import json
import logging
import os
import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar

from PIL import Image, ImageDraw, ImageFont
import io
import base64
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from browser_use.agent.gif import create_history_gif
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList, ToolCallingMethod, AgentState, AgentOutput
from browser_use.browser.views import BrowserState
from browser_use.controller.service import Controller
from browser_use.utils import time_execution_async
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext

from src.agent.custom_agent import CustomAgent
from src.agent.custom_message_manager import CustomMessageManager, CustomMessageManagerSettings
from src.agent.custom_views import CustomAgentOutput, CustomAgentStepInfo, CustomAgentState

logger = logging.getLogger(__name__)

Context = TypeVar('Context')

class StoryAgent(CustomAgent):
    """
    Agent that generates a story, creates images for each scene, and combines them into a GIF.
    """
    def __init__(
            self,
            task: str,
            llm: BaseChatModel,
            add_infos: str = "",
            # Optional parameters
            browser: Browser | None = None,
            browser_context: BrowserContext | None = None,
            controller: Controller[Context] = Controller(),
            # Initial agent run parameters
            sensitive_data: Optional[Dict[str, str]] = None,
            initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
            # Cloud Callbacks
            register_new_step_callback: Callable[[BrowserState, AgentOutput, int], Awaitable[None]] | None = None,
            register_done_callback: Callable[[AgentHistoryList], Awaitable[None]] | None = None,
            register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
            # Agent settings
            use_vision: bool = True,
            use_vision_for_planner: bool = False,
            save_conversation_path: Optional[str] = None,
            save_conversation_path_encoding: Optional[str] = 'utf-8',
            max_failures: int = 3,
            retry_delay: int = 10,
            system_prompt_class: Type[SystemPrompt] = SystemPrompt,
            agent_prompt_class: Type[AgentMessagePrompt] = AgentMessagePrompt,
            max_input_tokens: int = 128000,
            validate_output: bool = False,
            message_context: Optional[str] = None,
            generate_gif: bool | str = False,
            available_file_paths: Optional[list[str]] = None,
            include_attributes: list[str] = [
                'title',
                'type',
                'name',
                'role',
                'aria-label',
                'placeholder',
                'value',
                'alt',
                'aria-expanded',
                'data-date-format',
            ],
            max_actions_per_step: int = 10,
            tool_calling_method: Optional[ToolCallingMethod] = 'auto',
            page_extraction_llm: Optional[BaseChatModel] = None,
            planner_llm: Optional[BaseChatModel] = None,
            planner_interval: int = 1,  # Run planner every N steps
            # Inject state
            injected_agent_state: Optional[AgentState] = None,
            context: Context | None = None,
            # Story agent specific
            image_generation_model: str = "dall-e-3",
            image_generation_api_key: Optional[str] = None,
            save_story_path: Optional[str] = None,
    ):
        super().__init__(
            task=task,
            llm=llm,
            add_infos=add_infos,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            sensitive_data=sensitive_data,
            initial_actions=initial_actions,
            register_new_step_callback=register_new_step_callback,
            register_done_callback=register_done_callback,
            register_external_agent_status_raise_error_callback=register_external_agent_status_raise_error_callback,
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class,
            agent_prompt_class=agent_prompt_class,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            page_extraction_llm=page_extraction_llm,
            planner_llm=planner_llm,
            planner_interval=planner_interval,
            injected_agent_state=injected_agent_state,
            context=context,
        )
        
        self.image_generation_model = image_generation_model
        self.image_generation_api_key = image_generation_api_key or os.getenv("OPENAI_API_KEY")
        
        # Create base stories directory
        self.base_story_path = save_story_path or "story_output"
        os.makedirs(self.base_story_path, exist_ok=True)
        
        # Create a timestamped folder for this story
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        story_title = self._get_safe_title(task)
        self.save_story_path = os.path.join(self.base_story_path, f"{timestamp}_{story_title}")
        os.makedirs(self.save_story_path, exist_ok=True)
        
        self.story_images = []
        self.story_data = {}
        
        logger.info(f"Story will be saved to: {self.save_story_path}")
    
    def _get_safe_title(self, task: str) -> str:
        """Convert the task into a safe directory name"""
        # Take first 30 chars of task and replace unsafe chars with underscores
        import re
        # Take only alphanumeric characters and some safe ones, replace others with underscores
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', task)
        # Truncate to first 30 chars
        safe_title = safe_title[:30].strip('_')
        # If we end up with an empty string, use "story" as default
        if not safe_title:
            safe_title = "story"
        return safe_title
    
    async def generate_story(self):
        """Generate a story script using the LLM"""
        logger.info("Generating story script...")
        
        prompt = f"""
        Create a short story with 5-8 separate scenes or frames. 
        The story should be about: {self.task}
        
        First, define:
        1. Characters: List all main characters with detailed descriptions of their appearance, clothing, and distinguishing features.
        2. Style: Define a consistent visual style for the story (e.g., cartoon, photorealistic, watercolor, anime, etc.).
        3. Setting: Describe the overall setting's visual characteristics.
        
        Then create the story scenes.
        
        Format the response as a JSON object with:
        
        {{
          "style_guide": {{
            "art_style": "string",
            "color_palette": "string",
            "visual_theme": "string"
          }},
          "characters": [
            {{
              "name": "string",
              "description": "detailed visual description",
              "role": "string"
            }}
          ],
          "settings": [
            {{
              "name": "string",
              "description": "detailed visual description"
            }}
          ],
          "scenes": [
            {{
              "scene_number": number,
              "description": "string",
              "narration": "string",
              "characters_present": ["character names"]
            }}
          ]
        }}
        
        Make it a cohesive story with a beginning, middle, and end. Ensure all visuals can remain consistent throughout the story.
        """
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.agenerate([[messages[0]]])
        
        # Extract the JSON from the response
        content = response.generations[0][0].text
        try:
            # Try to extract JSON from the text if it's not already JSON formatted
            if not content.strip().startswith('[') and not content.strip().startswith('{'):
                import re
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                match = re.search(json_pattern, content)
                if match:
                    content = match.group(1)
                else:
                    # Try to find anything that looks like JSON
                    json_pattern = r'(\{[\s\S]*\})'
                    match = re.search(json_pattern, content)
                    if match:
                        content = match.group(1)
            
            story_data = json.loads(content)
            
            # Save the story script to a file
            script_path = os.path.join(self.save_story_path, "story_script.json")
            with open(script_path, 'w') as f:
                json.dump(story_data, f, indent=2)
                
            logger.info(f"Story script generated and saved to {script_path}")
            return story_data
        except Exception as e:
            logger.error(f"Failed to parse story script: {e}")
            logger.error(f"Raw response: {content}")
            raise
    
    async def generate_image_for_scene(self, scene):
        """Generate an image for a scene using an image generation service"""
        try:
            import openai
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=self.image_generation_api_key)
            
            # Create a detailed prompt that includes style and character consistency
            style_guide = getattr(self, 'story_data', {}).get('style_guide', {})
            characters = getattr(self, 'story_data', {}).get('characters', [])
            settings = getattr(self, 'story_data', {}).get('settings', [])
            
            # Extract characters present in this scene
            characters_present = scene.get('characters_present', [])
            character_descriptions = []
            
            for char_name in characters_present:
                # Find the character in our characters list
                for char in characters:
                    if char.get('name') == char_name:
                        character_descriptions.append(f"{char_name}: {char.get('description', '')}")
                        break
            
            # Build the prompt
            art_style = style_guide.get('art_style', '')
            color_palette = style_guide.get('color_palette', '')
            visual_theme = style_guide.get('visual_theme', '')
            
            # Create the style portion of the prompt
            style_prompt = f"Style: {art_style}. {color_palette}. {visual_theme}. "
            
            # Create the characters portion of the prompt
            characters_prompt = "Characters: " + "; ".join(character_descriptions) + ". " if character_descriptions else ""
            
            # Create the scene-specific prompt
            scene_prompt = scene.get('description', '')
            
            # Combine all parts into a final prompt
            final_prompt = f"{style_prompt}{characters_prompt}Scene: {scene_prompt}"
            
            logger.info(f"Generating image for scene {scene.get('scene_number', 0)}...")
            response = client.images.generate(
                model=self.image_generation_model,
                prompt=final_prompt,
                n=1,
                size="1024x1024"
            )
            
            # Get the image URL (ensure it's a string)
            image_url = response.data[0].url
            if image_url is None:
                raise ValueError("Image URL is None")
                
            # Download the image
            import requests
            image_response = requests.get(str(image_url))
            from PIL import Image
            image = Image.open(io.BytesIO(image_response.content))
            
            # Add scene description to the image
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to load a nice font, fallback to default if not available
            try:
                if os.path.exists("/System/Library/Fonts/Supplemental/Arial.ttf"):
                    # macOS path
                    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
                elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                    # Linux path
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
                elif os.path.exists("C:\\Windows\\Fonts\\arial.ttf"):
                    # Windows path
                    font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 36)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
                
            # Add a semi-transparent overlay at the bottom
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([(0, image.height - 200), (image.width, image.height)], 
                                  fill=(0, 0, 0, 180))
            
            # Add scene number
            scene_number_text = f"Scene {scene.get('scene_number', 0)}"
            draw.text((20, image.height - 180), scene_number_text, font=font, fill=(255, 255, 255))
            
            # Add scene description (wrapped to fit)
            wrapped_text = self._wrap_text(scene.get('description', ''), font, image.width - 40)
            y_position = image.height - 130
            for line in wrapped_text:
                draw.text((20, y_position), line, font=font, fill=(255, 255, 255))
                y_position += 40
            
            # Save the image to a file
            image_path = os.path.join(self.save_story_path, f"scene_{scene.get('scene_number', 0)}.png")
            image.save(image_path)
            
            logger.info(f"Image for scene {scene.get('scene_number', 0)} saved to {image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to generate image for scene {scene['scene_number']}: {e}")
            # Create a simple text image as a fallback
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            
            # Add scene number and description
            d.text((10, 10), f"Scene {scene['scene_number']}", fill=(0, 0, 0))
            
            # Wrap the description text
            wrapped_desc = self._wrap_text(scene['description'], ImageFont.load_default(), 1000)
            y = 50
            for line in wrapped_desc:
                d.text((10, y), line, fill=(0, 0, 0))
                y += 20
            
            image_path = os.path.join(self.save_story_path, f"scene_{scene['scene_number']}.png")
            img.save(image_path)
            logger.info(f"Fallback image for scene {scene['scene_number']} saved to {image_path}")
            return img
    
    def _wrap_text(self, text, font, max_width):
        """Helper function to wrap text to fit within a given width"""
        words = text.split()
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            # Check if adding this word would exceed the max width
            test_line = current_line + " " + word
            try:
                # For PIL.ImageFont objects that support getbbox or getsize
                if hasattr(font, 'getbbox'):
                    width = font.getbbox(test_line)[2]
                elif hasattr(font, 'getsize'):
                    width = font.getsize(test_line)[0]
                else:
                    # Fallback to a simple character count estimation
                    width = len(test_line) * 10
            except:
                # Fallback to a simple character count estimation
                width = len(test_line) * 10
                
            if width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
                
        lines.append(current_line)
        return lines
    
    async def create_story_gif(self, images, output_path=None):
        """Create a GIF from the story images"""
        if not output_path:
            output_path = os.path.join(self.save_story_path, "story.gif")
            
        logger.info(f"Creating story GIF at {output_path}...")
        
        # Save the first image as GIF and append the rest
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=3000,  # 3 seconds per frame
            loop=0  # Loop indefinitely
        )
        
        logger.info(f"Story GIF created at {output_path}")
        return output_path
    
    async def run(self, max_steps: int = 100):
        """Run the story generation process"""
        try:
            logger.info(f"Starting story generation for task: {self.task}")
            
            # Step 1: Generate story script
            self.story_data = await self.generate_story()
            
            # Step 2: Generate images for each scene
            images = []
            for scene in self.story_data['scenes']:
                image = await self.generate_image_for_scene(scene)
                images.append(image)
                self.story_images.append(image)
            
            # Step 3: Create the story GIF
            gif_path = await self.create_story_gif(images)
            
            # Return a simple result dictionary
            result = {
                "success": True,
                "task": self.task,
                "scenes": len(self.story_data['scenes']),
                "gif_path": gif_path,
                "script_path": os.path.join(self.save_story_path, "story_script.json"),
                "error": None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during story generation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error result
            return {
                "success": False,
                "task": self.task,
                "error": str(e),
                "gif_path": None,
                "script_path": None
            } 