import json
import logging
import os
import asyncio
import time
import random
import io
import numpy as np
import requests
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union

from PIL import Image, ImageDraw, ImageFont

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
from src.utils.utils import create_video_from_images

import imageio
from datetime import datetime

logger = logging.getLogger(__name__)

Context = TypeVar('Context')

class StableDiffusionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        logging.getLogger(__name__).info(f"Initialized StableDiffusionClient with base URL: {base_url}")

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution"
    ) -> Image.Image:
        """
        Generate an image using Stable Diffusion v1.5
        
        Args:
            prompt: Text description of the desired image
            num_inference_steps: Number of denoising steps (higher = better quality but slower)
            guidance_scale: How closely to follow the prompt (higher = more faithful but less creative)
            negative_prompt: Text description of what to avoid in the image
            
        Returns:
            PIL Image object
        """
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt
        }
        
        logger = logging.getLogger(__name__)
        logger.info(f"Sending request to Stable Diffusion server with prompt: {prompt[:50]}...")
        
        try:
            response = requests.post(f"{self.base_url}/predict", json=payload)
            
            # Log response details for debugging
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response content type: {response.headers.get('content-type', 'unknown')}")
            logger.info(f"Response content length: {len(response.content)}")
            
            if response.status_code != 200:
                error_msg = f"Error generating image: {response.status_code} - {response.text[:200]}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Check if response contains image data
            if not response.content or len(response.content) < 100:
                error_msg = f"Response doesn't contain valid image data. Content length: {len(response.content)}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            # Convert raw bytes to PIL Image
            try:
                return Image.open(io.BytesIO(response.content))
            except Exception as e:
                error_msg = f"Failed to create PIL Image from response content: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except requests.RequestException as e:
            error_msg = f"Request to Stable Diffusion server failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

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
            injected_agent_state: Optional[Union[AgentState, CustomAgentState]] = None,
            context: Context | None = None,
            # Story agent specific
            image_generation_model: str = "dall-e-3",
            image_generation_api_key: Optional[str] = None,
            save_story_path: Optional[str] = None,
            # Additional consistency options
            use_image_seed: bool = True,
            generate_video: bool = True,
            video_framerate: int = 2,
            # Frame duration control
            gif_frame_duration: float = 3.0,  # Duration in seconds for each frame in GIF
            video_frame_duration: float = 0.5,  # Duration in seconds for each frame in video
            variable_durations: Optional[List[float]] = None,  # Optional list of durations for each frame
            # Local image generation options
            use_local_generation: bool = False,
            local_generation_url: str = "http://localhost:8000",
            local_generation_steps: int = 20,
            local_generation_guidance_scale: float = 7.5,
            local_generation_negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution"
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
            injected_agent_state=injected_agent_state if not isinstance(injected_agent_state, CustomAgentState) else AgentState(),
            context=context,
        )
        
        self.image_generation_model = image_generation_model
        self.image_generation_api_key = image_generation_api_key or os.getenv("OPENAI_API_KEY")
        
        # Create base stories directory
        self.base_story_path = save_story_path or "story_output"
        os.makedirs(self.base_story_path, exist_ok=True)
        
        # Create a timestamped folder for this story
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        story_title = self._get_safe_title(task)
        self.save_story_path = os.path.join(self.base_story_path, f"{timestamp}_{story_title}")
        os.makedirs(self.save_story_path, exist_ok=True)
        
        self.story_images = []
        self.story_data = {}
        
        # Consistency options
        self.use_image_seed = use_image_seed
        self.image_seed = self._generate_seed() if use_image_seed else None
        
        logger.info(f"Story will be saved to: {self.save_story_path}")
        
        # Additional story agent specific
        self.generate_video = generate_video
        self.video_framerate = video_framerate
        
        # Frame duration control
        self.gif_frame_duration = gif_frame_duration
        self.video_frame_duration = video_frame_duration
        self.variable_durations = variable_durations
        
        # Story data
        self.story_id = None
        self.story_folder = None
        self.story_frames = []
        self.story_script = {}
        self.seed = None if not use_image_seed else str(datetime.now().timestamp())
        
        # Stable Diffusion local generation options
        self.use_local_generation = use_local_generation
        self.local_generation_url = local_generation_url
        self.local_generation_steps = local_generation_steps
        self.local_generation_guidance_scale = local_generation_guidance_scale
        self.local_generation_negative_prompt = local_generation_negative_prompt
        
        if use_local_generation:
            try:
                self.sd_client = StableDiffusionClient(base_url=local_generation_url)
                # Test the connection
                logger.info(f"Testing connection to local Stable Diffusion server at {local_generation_url}...")
                response = requests.get(f"{local_generation_url}/docs", timeout=5)
                if response.status_code == 200:
                    logger.info("Successfully connected to local Stable Diffusion server")
                else:
                    logger.warning(f"Local Stable Diffusion server responded with status code {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to initialize local Stable Diffusion client: {e}")
                logger.warning("Falling back to OpenAI for image generation")
                self.use_local_generation = False
        
        # Flag for stopping
        self.stopped = False
    
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
    
    def _generate_seed(self) -> int:
        """Generate a consistent seed for image generation"""
        import random
        # Generate a stable seed within OpenAI's accepted range
        return random.randint(1, 4294967295)  # Max 32-bit integer

    async def generate_story(self):
        """Generate a story script using the LLM"""
        logger.info("Generating story script...")
        
        # Check if the task contains Chinese characters
        import re
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', self.task))
        
        if has_chinese:
            prompt = f"""
            请创建一个有5-8个独立场景或画面的短篇故事。
            故事主题是：{self.task}
            
            首先，请定义：
            1. 角色：列出所有主要角色，详细描述他们的外表、服装和显著特征。
            2. 风格：定义故事的一致视觉风格（例如，卡通、写实、水彩、动漫等）。
            3. 场景：描述整体场景的视觉特征。
            
            然后创建故事场景。
            
            请以JSON对象格式回复，结构如下：
            
            {{
              "style_guide": {{
                "art_style": "string",
                "color_palette": "string",
                "visual_theme": "string"
              }},
              "characters": [
                {{
                  "name": "string",
                  "description": "详细的视觉描述",
                  "role": "string"
                }}
              ],
              "settings": [
                {{
                  "name": "string",
                  "description": "详细的视觉描述"
                }}
              ],
              "scenes": [
                {{
                  "scene_number": number,
                  "description": "string",
                  "narration": "string",
                  "characters_present": ["角色名称"]
                }}
              ]
            }}
            
            请确保故事有一个完整的开始、中间和结尾。确保所有视觉元素在整个故事中保持一致性。
            确保您的JSON有效且格式正确！
            """
        else:
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
            Make sure your JSON is valid and properly formatted!
            """
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.agenerate([[messages[0]]])
        
        # Extract the JSON from the response
        content = response.generations[0][0].text
        try:
            # Try to extract JSON from the text if it's not already JSON formatted
            if not content.strip().startswith('{'):
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
            
            # Try to parse the JSON
            try:
                story_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {e}. Attempting to repair JSON...")
                
                # Try to repair the JSON
                try:
                    from json_repair import repair_json
                    repaired_content = repair_json(content)
                    story_data = json.loads(repaired_content)
                    logger.info("Successfully repaired and parsed JSON")
                except Exception as repair_error:
                    logger.error(f"JSON repair failed: {repair_error}")
                    
                    # Last resort: manual repair of common issues
                    content_fixed = content.replace("'", '"')  # Replace single quotes with double quotes
                    content_fixed = re.sub(r',\s*}', '}', content_fixed)  # Remove trailing commas in objects
                    content_fixed = re.sub(r',\s*]', ']', content_fixed)  # Remove trailing commas in arrays
                    
                    try:
                        story_data = json.loads(content_fixed)
                        logger.info("Successfully parsed JSON after manual fixes")
                    except json.JSONDecodeError as final_error:
                        # If all parsing attempts fail, create a basic structure
                        logger.error(f"All JSON parsing attempts failed. Creating basic structure: {final_error}")
                        story_data = {
                            "style_guide": {"art_style": "cartoon", "color_palette": "vibrant", "visual_theme": "adventure"},
                            "characters": [{"name": "Character", "description": "A character", "role": "Main"}],
                            "settings": [{"name": "Setting", "description": "A place"}],
                            "scenes": [
                                {"scene_number": 1, "description": self.task, "narration": self.task, "characters_present": ["Character"]}
                            ]
                        }
            
            # Save the story script to a file
            script_path = os.path.join(self.save_story_path, "story_script.json")
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Story script generated and saved to {script_path}")
            return story_data
        except Exception as e:
            logger.error(f"Failed to process story script: {e}")
            logger.error(f"Raw response: {content}")
            
            # Create a basic structure as fallback
            fallback_data = {
                "style_guide": {"art_style": "cartoon", "color_palette": "vibrant", "visual_theme": "adventure"},
                "characters": [{"name": "Character", "description": "A character", "role": "Main"}],
                "settings": [{"name": "Setting", "description": "A place"}],
                "scenes": [
                    {"scene_number": 1, "description": self.task, "narration": self.task, "characters_present": ["Character"]}
                ]
            }
            
            # Save the fallback script
            script_path = os.path.join(self.save_story_path, "story_script.json")
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_data, f, indent=2, ensure_ascii=False)
            
            logger.warning(f"Using fallback story structure due to parsing error")
            return fallback_data
    
    async def generate_image_for_scene(self, scene):
        """Generate an image for a scene using an image generation service"""
        try:
            # Build the prompt
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
            
            # Check if we're dealing with Chinese text
            import re
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', self.task))
            
            # Create the style portion of the prompt
            if has_chinese:
                style_prompt = f"风格: {art_style}. {color_palette}. {visual_theme}. "
            else:
                style_prompt = f"Style: {art_style}. {color_palette}. {visual_theme}. "
            
            # Create the characters portion of the prompt
            if has_chinese:
                characters_prompt = "角色: " + "; ".join(character_descriptions) + ". " if character_descriptions else ""
            else:
                characters_prompt = "Characters: " + "; ".join(character_descriptions) + ". " if character_descriptions else ""
            
            # Create the scene-specific prompt
            scene_prompt = scene.get('description', '')
            
            # If using seed, add it to the prompt for consistency without relying on API parameter
            seed_prompt = ""
            if self.use_image_seed and self.image_seed is not None:
                seed_prompt = f" Seed: {self.image_seed}."
            
            # Combine all parts into a final prompt
            if has_chinese:
                final_prompt = f"{style_prompt}{characters_prompt}场景: {scene_prompt}{seed_prompt}"
            else:
                final_prompt = f"{style_prompt}{characters_prompt}Scene: {scene_prompt}{seed_prompt}"
            
            # Add a consistency reminder
            if character_descriptions:
                if has_chinese:
                    final_prompt += " 在整个故事中保持角色外观的一致性。"
                else:
                    final_prompt += " Maintain consistent character appearances throughout the story."
            
            logger.info(f"Generating image for scene {scene.get('scene_number', 0)}...")
            
            # Generate image using either local Stable Diffusion or OpenAI API
            if self.use_local_generation:
                logger.info(f"Using local Stable Diffusion for scene {scene.get('scene_number', 0)}")
                try:
                    image = self.sd_client.generate_image(
                        prompt=final_prompt,
                        num_inference_steps=self.local_generation_steps,
                        guidance_scale=self.local_generation_guidance_scale,
                        negative_prompt=self.local_generation_negative_prompt
                    )
                    # Verify the image was created correctly
                    from PIL import Image as PILImage
                    if not image or not isinstance(image, PILImage.Image):
                        raise ValueError(f"Invalid image returned from local generation: {type(image)}")
                    
                    logger.info(f"Successfully generated image locally for scene {scene.get('scene_number', 0)}")
                except Exception as e:
                    logger.error(f"Local image generation failed: {e}")
                    logger.info("Creating a fallback image")
                    
                    # Create a simple fallback image with error text
                    from PIL import Image as PILImage
                    from PIL import ImageDraw
                    
                    fallback_img = PILImage.new('RGB', (1024, 1024), color=(255, 255, 255))
                    draw = ImageDraw.Draw(fallback_img)
                    draw.text((10, 10), f"Error: {str(e)[:100]}", fill=(0, 0, 0))
                    draw.text((10, 40), f"Scene: {scene.get('scene_number', 0)}", fill=(0, 0, 0))
                    draw.text((10, 70), f"Prompt: {final_prompt[:100]}...", fill=(0, 0, 0))
                    
                    image = fallback_img
                    logger.warning(f"Using fallback image for scene {scene.get('scene_number', 0)}")
            else:
                # Use OpenAI for image generation
                import openai
                
                # Initialize OpenAI client (or midjourney client based on model selection)
                client = openai.OpenAI(api_key=self.image_generation_api_key)
                
                # Make the image generation call without seed parameter to avoid compatibility issues
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
            from PIL import Image, ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to load a font that supports multi-language characters (especially Chinese)
            try:
                # Look for fonts with Chinese character support
                font = None
                
                # macOS Chinese font paths
                mac_chinese_fonts = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Light.ttc",
                    "/Library/Fonts/Arial Unicode.ttf"
                ]
                
                # Linux Chinese font paths
                linux_chinese_fonts = [
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
                ]
                
                # Windows Chinese font paths
                windows_chinese_fonts = [
                    "C:\\Windows\\Fonts\\msyh.ttc",  # Microsoft YaHei
                    "C:\\Windows\\Fonts\\simsun.ttc",  # SimSun
                    "C:\\Windows\\Fonts\\simhei.ttf"   # SimHei
                ]
                
                # Try each font path until we find one that exists
                for font_path in mac_chinese_fonts + linux_chinese_fonts + windows_chinese_fonts:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, 36)
                        logger.info(f"Using font: {font_path}")
                        break
                
                # Fallback to standard fonts if no Chinese fonts found
                if font is None:
                    if os.path.exists("/System/Library/Fonts/Supplemental/Arial.ttf"):
                        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
                    elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
                    elif os.path.exists("C:\\Windows\\Fonts\\arial.ttf"):
                        font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 36)
                    else:
                        font = ImageFont.load_default()
            except Exception as e:
                logger.warning(f"Could not load font: {e}")
                font = ImageFont.load_default()
                
            # Add a semi-transparent overlay at the bottom
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([(0, image.height - 200), (image.width, image.height)], 
                                  fill=(0, 0, 0, 180))
            
            # Add scene number
            if has_chinese:
                scene_number_text = f"场景 {scene.get('scene_number', 0)}"
            else:
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
            logger.error(f"Failed to generate image for scene {scene.get('scene_number', 0)}: {e}")
            try:
                # Create a simple text image as a fallback
                # Import PIL modules here with explicit aliases to avoid scoping issues
                from PIL import Image as PILImage
                from PIL import ImageDraw
                from PIL import ImageFont
                
                img = PILImage.new('RGB', (1024, 1024), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                
                # Add scene number and description
                d.text((10, 10), f"Scene {scene.get('scene_number', 0)}", fill=(0, 0, 0))
                
                # Get default font
                default_font = ImageFont.load_default()
                
                # Wrap the description text
                wrapped_desc = self._wrap_text(scene.get('description', ''), default_font, 1000)
                y = 50
                for line in wrapped_desc:
                    d.text((10, y), line, fill=(0, 0, 0))
                    y += 20
                
                image_path = os.path.join(self.save_story_path, f"scene_{scene.get('scene_number', 0)}.png")
                img.save(image_path)
                logger.info(f"Fallback image for scene {scene.get('scene_number', 0)} saved to {image_path}")
                return img
            except Exception as inner_e:
                # If even the fallback fails, log and return a minimal image
                logger.error(f"Failed to create fallback image: {inner_e}")
                from PIL import Image as PILImage
                minimal_img = PILImage.new('RGB', (512, 512), color=(200, 200, 200))
                minimal_path = os.path.join(self.save_story_path, f"scene_{scene.get('scene_number', 0)}_minimal.png")
                minimal_img.save(minimal_path)
                return minimal_img
    
    def _wrap_text(self, text, font, max_width):
        """Helper function to wrap text to fit within a given width, supporting both English and Chinese"""
        if not text:
            return []
            
        # For Chinese text, we need to handle characters individually
        # Check if the text contains Chinese characters
        import re
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        if has_chinese:
            # For Chinese text, handle character by character
            lines = []
            current_line = ""
            
            for char in text:
                # Try adding the next character
                test_line = current_line + char
                
                # Check if width exceeds max_width
                try:
                    if hasattr(font, 'getbbox'):
                        width = font.getbbox(test_line)[2]
                    elif hasattr(font, 'getsize'):
                        width = font.getsize(test_line)[0]
                    else:
                        # Fallback estimation
                        width = len(test_line) * 20  # Wider estimate for Chinese
                except:
                    width = len(test_line) * 20
                
                if width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = char
            
            # Add the last line
            if current_line:
                lines.append(current_line)
                
            return lines
        else:
            # Original word-by-word wrapping for non-Chinese text
            words = text.split()
            if not words:
                return []
                
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
        
        # Use variable durations if provided
        if self.variable_durations and len(self.variable_durations) >= len(images):
            # Convert seconds to milliseconds
            durations = [int(d * 1000) for d in self.variable_durations[:len(images)]]
            logger.info(f"Using variable frame durations: {durations}")
        else:
            # Use fixed duration (converted from seconds to milliseconds)
            durations = int(self.gif_frame_duration * 1000)
            logger.info(f"Using fixed frame duration: {durations}ms")
        
        # Save the first image as GIF and append the rest
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=durations,  # Use the durations (fixed or variable)
            loop=0  # Loop indefinitely
        )
        
        logger.info(f"Story GIF created at {output_path}")
        return output_path
    
    async def generate_output_files(self):
        """Generate the output files: GIF and optionally MP4 video"""
        
        # Ensure we have frames to create outputs
        if not self.story_images:
            logging.error("No frames to generate outputs from")
            return None, None
            
        # Create GIF from frames
        gif_path = os.path.join(self.save_story_path, "story.gif")
        try:
            # Convert PIL images to numpy arrays for imageio if needed
            np_images = []
            for img in self.story_images:
                if hasattr(img, 'convert'):  # If it's a PIL Image
                    np_img = np.array(img.convert('RGB'))
                    np_images.append(np_img)
                else:  # If it's already a numpy array
                    np_images.append(img)
                
            # Use variable durations if provided, otherwise use fixed duration
            if self.variable_durations and len(self.variable_durations) >= len(np_images):
                # imageio uses seconds
                durations = self.variable_durations[:len(np_images)]
                logger.info(f"Using variable durations for GIF: {durations}")
                imageio.mimsave(gif_path, np_images, duration=durations, loop=0)
            else:
                # Use fixed duration in seconds
                logger.info(f"Using fixed duration for GIF: {self.gif_frame_duration}s")
                imageio.mimsave(gif_path, np_images, duration=self.gif_frame_duration, loop=0)
            
            logging.info(f"Generated GIF at {gif_path}")
        except Exception as e:
            import traceback
            logging.error(f"Error generating GIF: {str(e)}\n{traceback.format_exc()}")
            gif_path = None
        
        # Create MP4 video if enabled
        video_path = None
        if self.generate_video:
            try:
                # Save individual frames as images first
                frames_folder = os.path.join(self.save_story_path, "frames")
                os.makedirs(frames_folder, exist_ok=True)
                
                # For variable frame timing, we need to create multiple copies of frames
                if self.variable_durations and len(self.variable_durations) >= len(self.story_images):
                    frame_index = 0
                    for i, img in enumerate(self.story_images):
                        # Calculate how many copies to make based on frame duration
                        # Convert duration to equivalent number of frames at the given framerate
                        # 1 second at 24fps = 24 frames
                        frame_count = max(1, int(self.variable_durations[i] * self.video_framerate))
                        
                        # Save multiple copies of the same frame
                        for j in range(frame_count):
                            frame_path = os.path.join(frames_folder, f"frame_{frame_index:04d}.jpg")
                            frame_index += 1
                            if hasattr(img, 'save'):  # If it's a PIL Image
                                img.save(frame_path)
                            else:  # If it's a numpy array
                                imageio.imwrite(frame_path, img)
                else:
                    # For fixed duration, save each frame the appropriate number of times
                    frame_copies = max(1, int(self.video_frame_duration * self.video_framerate))
                    logger.info(f"Creating {frame_copies} copies of each frame for video")
                    
                    frame_index = 0
                    for img in self.story_images:
                        for j in range(frame_copies):
                            frame_path = os.path.join(frames_folder, f"frame_{frame_index:04d}.jpg")
                            frame_index += 1
                            if hasattr(img, 'save'):  # If it's a PIL Image
                                img.save(frame_path)
                            else:  # If it's a numpy array
                                imageio.imwrite(frame_path, img)
                
                # Generate the video
                video_path = os.path.join(self.save_story_path, "story.mp4")
                video_success = create_video_from_images(
                    frames_folder, 
                    video_path, 
                    framerate=self.video_framerate,
                    extension="jpg"
                )
                
                if not video_success:
                    logging.error("Failed to create video")
                    video_path = None
            except Exception as e:
                import traceback
                logging.error(f"Error generating video: {str(e)}\n{traceback.format_exc()}")
                video_path = None
        
        return gif_path, video_path
    
    async def run(self, max_steps: int = 100):
        """Run the story generation process"""
        if self.stopped:
            return {"success": False, "error": "Agent was stopped"}
            
        try:
            logger.info(f"Starting story generation for task: {self.task}")
            
            # Log which image generation method we're using
            if self.use_local_generation:
                logger.info(f"Using local Stable Diffusion server at {self.local_generation_url}")
            else:
                logger.info(f"Using OpenAI {self.image_generation_model} for image generation")
            
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
            
            # Step 4: Generate output files
            output_gif_path, output_video_path = await self.generate_output_files()
            
            # Return a simple result dictionary
            result = {
                "success": True,
                "task": self.task,
                "scenes": len(self.story_data['scenes']),
                "gif_path": output_gif_path,
                "video_path": output_video_path,
                "script_path": os.path.join(self.save_story_path, "story_script.json"),
                "image_generation_method": "local_stable_diffusion" if self.use_local_generation else f"openai_{self.image_generation_model}",
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
                "video_path": None,
                "script_path": None
            }
    
    def stop(self):
        self.stopped = True 