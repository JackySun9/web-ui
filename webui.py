import pdb
import logging
import json

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

import gradio as gr
import inspect
from functools import wraps

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot, MissingAPIKeyError
from src.utils import utils

# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()

# webui config
webui_config_manager = utils.ConfigManager()

# Get model names from utils
llm_models = utils.model_names if hasattr(utils, 'model_names') else {}


def scan_and_register_components(blocks):
    """扫描一个 Blocks 对象并注册其中的所有交互式组件，但不包括按钮"""
    global webui_config_manager

    def traverse_blocks(block, prefix=""):
        registered = 0

        # 处理 Blocks 自身的组件
        if hasattr(block, "children"):
            for i, child in enumerate(block.children):
                if isinstance(child, gr.components.Component):
                    # 排除按钮 (Button) 组件
                    if getattr(child, "interactive", False) and not isinstance(child, gr.Button):
                        name = f"{prefix}component_{i}"
                        if hasattr(child, "label") and child.label:
                            # 使用标签作为名称的一部分
                            label = child.label
                            name = f"{prefix}{label}"
                        logger.debug(f"Registering component: {name}")
                        webui_config_manager.register_component(name, child)
                        registered += 1
                elif hasattr(child, "children"):
                    # 递归处理嵌套的 Blocks
                    new_prefix = f"{prefix}block_{i}_"
                    registered += traverse_blocks(child, new_prefix)

        return registered

    total = traverse_blocks(blocks)
    logger.info(f"Total registered components: {total}")


def save_current_config():
    return webui_config_manager.save_current_config()


def update_ui_from_config(config_file):
    return webui_config_manager.update_ui_from_config(config_file)


def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text

    import re

    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)

    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)

    return result


async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent

    try:
        if _global_agent is not None:
            # Request stop
            _global_agent.stop()
        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def stop_research_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (  # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        elif agent_type == "deepsite":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_deepsite_agent(
                llm=llm,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        # latest_video = None
        # if save_recording_path:
        #     new_videos = set(
        #         glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
        #         + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
        #     )
        #     if new_videos - existing_videos:
        #         latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )

    except MissingAPIKeyError as e:
        logger.error(str(e))
        raise gr.Error(str(e), print_exception=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',  # final_result
            errors,  # errors
            '',  # model_actions
            '',  # model_thoughts
            None,  # latest_video
            None,  # history_file
            None,  # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None


async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp
        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)

            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        # if chrome_cdp not empty string nor None
        if (_global_browser is None) or (cdp_url and cdp_url != "" and cdp_url != None):
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None or (chrome_cdp and cdp_url != "" and cdp_url != None):
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        # Create and run agent
        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None


async def run_deepsite_agent(
        llm,
        task,
        max_steps=10,
        use_vision=True,
        max_actions_per_step=10,
        tool_calling_method="auto",
        max_input_tokens=128000
):
    try:
        print(f"Starting run_deepsite_agent with task: {task}")
        from datetime import datetime
        global _global_agent
        
        # Define the system prompt for website generation
        system_prompt = {
            "role": "system",
            "content": "ONLY USE HTML, CSS AND JAVASCRIPT. If you want to use ICON make sure to import the library first. Try to create the best UI possible by using only HTML, CSS and JAVASCRIPT. Use as much as you can TailwindCSS for the CSS, if you can't do something with TailwindCSS, then use custom CSS (make sure to import <script src=\"https://cdn.tailwindcss.com\"></script> in the head). Also, try to ellaborate as much as you can, to create something unique. ALWAYS GIVE THE RESPONSE INTO A SINGLE HTML FILE"
        }
        
        # Create a simple agent that just needs to generate HTML without browser interactions
        from langchain_core.messages import SystemMessage, HumanMessage
        
        print("Creating messages structure")
        # Create messages structure
        messages = [
            SystemMessage(content=system_prompt["content"]),
            HumanMessage(content=task)
        ]
        
        print("Invoking LLM model")
        # Invoke the LLM directly to generate the website content
        try:
            response = llm.invoke(messages)
            print("LLM response received successfully")
        except Exception as llm_error:
            print(f"Error invoking LLM: {str(llm_error)}")
            raise llm_error
        
        # Extract the HTML content from the response
        html_content = response.content
        
        # Clean up the code if it's wrapped in markdown code blocks
        if "```html" in html_content:
            html_content = html_content.split("```html")[1].split("```")[0].strip()
        elif "```" in html_content:
            html_content = html_content.split("```")[1].split("```")[0].strip()
            
        # Save the HTML to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("tmp", "deepsite")
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, f"website_{timestamp}.html")
        print(f"Saving HTML content to file: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print("Successfully completed run_deepsite_agent")
        # Return the same structure as other agent functions
        return html_content, "", "", "", None, file_path
        
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        print(f"Error in run_deepsite_agent: {error_details}")
        return '', error_details, '', '', None, None

async def run_story_agent(
        llm,
        task,
        image_generation_model="dall-e-3",
        image_generation_api_key=None,
        save_story_path=None,
        max_steps=10,
        use_image_seed=True,
        gif_frame_duration=3.0,
        video_frame_duration=0.5,
        video_framerate=2,
        variable_durations=None
):
    try:
        global _global_agent
        
        from src.agent.story_agent import StoryAgent
        from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
        
        # Create and run agent
        _global_agent = StoryAgent(
            task=task,
            llm=llm,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            image_generation_model=image_generation_model,
            image_generation_api_key=image_generation_api_key,
            save_story_path=save_story_path,
            use_image_seed=use_image_seed,
            generate_video=True,
            video_framerate=video_framerate,
            gif_frame_duration=gif_frame_duration,
            video_frame_duration=video_frame_duration,
            variable_durations=variable_durations
        )
        
        # Run the agent
        result = await _global_agent.run(max_steps=max_steps)
        
        # Check if generation was successful
        if result.get("success", False):
            gif_path = result.get("gif_path")
            video_path = result.get("video_path")
            script_path = result.get("script_path")
            
            # Read the script content
            script_content = ""
            if script_path and os.path.exists(script_path):
                try:
                    with open(script_path, 'r', encoding='utf-8') as f:
                        script_content = f.read()
                except Exception as e:
                    logger.error(f"Error reading script file: {e}")
            
            return gif_path, '', script_content, '', video_path, script_path
        else:
            # Return the error message
            return '', result.get("error", "Story generation failed"), '', '', None, None
            
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        return '', error_details, '', '', None, None
    finally:
        _global_agent = None


async def run_with_stream(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    global _global_agent

    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            chrome_cdp=chrome_cdp,
            max_input_tokens=max_input_tokens
        )
        # Add HTML content at the start of the result array
        yield [gr.update(visible=False)] + list(result)
    else:
        try:
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_num_ctx=llm_num_ctx,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method,
                    chrome_cdp=chrome_cdp,
                    max_input_tokens=max_input_tokens
                )
            )

            # Initialize values for streaming
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            recording_gif = trace = history_file = None

            # Periodically update the stream while the agent task is running
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception as e:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent and _global_agent.state.stopped:
                    yield [
                        gr.HTML(value=html_content, visible=True),
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        recording_gif,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),  # stop_button
                        gr.update(interactive=False),  # run_button
                    ]
                    break
                else:
                    yield [
                        gr.HTML(value=html_content, visible=True),
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        recording_gif,
                        trace,
                        history_file,
                        gr.update(),  # Re-enable stop button
                        gr.update()  # Re-enable run button
                    ]
                await asyncio.sleep(0.1)

            # Once the agent task completes, get the results
            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, recording_gif, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                recording_gif = trace = history_file = None

            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                gr.HTML(value=html_content, visible=True),
                final_result,
                errors,
                model_actions,
                model_thoughts,
                recording_gif,
                trace,
                history_file,
                stop_button,
                run_button
            ]

        except Exception as e:
            import traceback
            yield [
                gr.HTML(
                    value=f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                    visible=True),
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True)  # Re-enable run button
            ]


# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}


async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None


async def run_deep_search(research_task, max_search_iteration_input, max_query_per_iter_input, llm_provider,
                          llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision,
                          use_own_browser, headless, chrome_cdp):
    from src.utils.deep_research import deep_research
    global _global_agent_state

    # Clear any previous stop request
    _global_agent_state.clear_stop()

    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        num_ctx=llm_num_ctx,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    markdown_content, file_path = await deep_research(research_task, llm, _global_agent_state,
                                                      max_search_iterations=max_search_iteration_input,
                                                      max_query_num=max_query_per_iter_input,
                                                      use_vision=use_vision,
                                                      headless=headless,
                                                      use_own_browser=use_own_browser,
                                                      chrome_cdp=chrome_cdp
                                                      )

    return markdown_content, file_path, gr.update(value="Stop", interactive=True), gr.update(interactive=True)


async def run_website_test(test_type, target_url, test_config, llm_provider, llm_model_name, 
                        llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, 
                        use_vision, use_own_browser, headless, disable_security, window_w, window_h, chrome_cdp, keep_browser_open, max_steps=25):
    """Run a website test with specified parameters using CustomAgent for consistency"""
    
    try:
        # To avoid import errors
        import os
        from datetime import datetime
        
        global _global_browser, _global_browser_context, _global_agent
        
        # Initialize LLM
        logger.info(f"Initializing LLM for website testing: {llm_provider}/{llm_model_name}")
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        
        # Ensure test_config is a dictionary
        if test_config is None:
            test_config = {}
            
        if not target_url:
            raise ValueError("Please provide a valid URL to test")
            
        # Add http:// if missing from URL
        if not target_url.startswith("http"):
            target_url = "https://" + target_url
        
        logger.info(f"Starting website test: {test_type} for {target_url}")
        
        # Get scope from config
        scope = test_config.get("scope", "limited navigation")
        scope_instructions = ""
        
        if scope == "homepage only":
            scope_instructions = "Focus only on the homepage. Do not navigate to other pages."
        elif scope == "limited navigation":
            scope_instructions = "Test the homepage thoroughly and check a few important links, but don't go deeper than 2 levels."
        elif scope == "thorough exploration":
            scope_instructions = "Explore the website thoroughly, following important links and examining key sections in detail."
        
        # Create task description based on test type
        if test_type == "functionality":
            task = f"""Test the website {target_url} for functionality issues.
{scope_instructions}

Follow this testing methodology for efficiency:
1. First, analyze the page layout and structure to identify key areas to test
2. Prioritize testing essential functionality based on the website's purpose
3. Document specific issues rather than general observations
4. Take screenshots of problems you encounter

Tasks:
{" - Analyze links: Click on main navigation links and check if they work correctly" if test_config.get("check_links", False) else ""}
{" - Check forms: Find and analyze forms, test with sample inputs if possible" if test_config.get("check_forms", False) else ""}
{" - Evaluate navigation: Explore the site navigation and assess its logical structure" if test_config.get("check_navigation", False) else ""}

Provide a detailed analysis with specific findings and take screenshots of any issues.
Your final answer should be formatted in Markdown with a summary section and a detailed findings section.
"""
            
        elif test_type == "accessibility":
            wcag_level = test_config.get("wcag_level", "AA")
            task = f"""Test the website {target_url} for WCAG {wcag_level} accessibility compliance.
{scope_instructions}

Follow this testing methodology for efficiency:
1. First scan for critical accessibility issues (missing alt text, keyboard navigation issues)
2. Check color contrast on important elements like buttons, links and text
3. Test keyboard navigation for key interactive elements
4. Document specific examples of issues rather than general statements

Tasks:
{" - Analyze color contrast: Identify elements with poor contrast ratios" if test_config.get("check_contrast", False) else ""}
{" - Check alt text: Find images and verify they have appropriate alt text" if test_config.get("check_alt_text", False) else ""}
{" - Evaluate ARIA attributes: Check ARIA usage and accessibility structure" if test_config.get("check_aria", False) else ""}

Provide a detailed accessibility audit according to WCAG {wcag_level} standards.
Your final answer should be formatted in Markdown with a summary section and a detailed findings section.
"""
            
        elif test_type == "performance":
            device_types = test_config.get("device_types", ["Desktop"])
            device_instructions = ""
            if "Mobile" in device_types:
                device_instructions += " View the site on a mobile device by resizing the browser window to a small size (e.g. 375x667)."
            if "Tablet" in device_types:
                device_instructions += " View the site on a tablet by resizing the browser window to a medium size (e.g. 768x1024)."
                
            task = f"""Test the website {target_url} for performance optimization opportunities.{device_instructions}
{scope_instructions}

Follow this testing methodology for efficiency:
1. First identify large media elements (images, videos) that might slow loading
2. Check for render-blocking resources like scripts and stylesheets
3. Test responsive behavior by resizing the browser window
4. Focus on specific examples of issues, not general observations

Tasks:
{" - Analyze page structure: Identify elements that might slow down loading" if test_config.get("check_loading", False) else ""}
{" - Check resources: Look for resource-heavy elements like large images, videos, scripts" if test_config.get("check_resources", False) else ""}
{" - Evaluate responsiveness: Test how the site behaves on different screen sizes" if test_config.get("check_responsiveness", False) else ""}

Provide a detailed performance analysis with specific findings and improvement recommendations.
Your final answer should be formatted in Markdown with a summary section and a detailed findings section.
"""
        else:
            task = f"""Analyze the website {target_url} and provide insights about its design, functionality, and user experience.
{scope_instructions}

Follow this testing methodology for efficiency:
1. First get an overview of the site's purpose and main features
2. Identify key user flows and test them
3. Document specific observations with examples
4. Take screenshots to illustrate your findings

Your final answer should be formatted in Markdown with a summary section and a detailed findings section.
"""

        # Set up paths for recording files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_recording_path = os.path.join("tmp", "record_videos")
        save_agent_history_path = os.path.join("tmp", "agent_history")
        save_trace_path = os.path.join("tmp", "traces")
        
        # Ensure directories exist
        for path in [save_recording_path, save_agent_history_path, save_trace_path]:
            os.makedirs(path, exist_ok=True)
            
        # Configuration for browser
        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp
        
        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None
            
        controller = CustomController()
        
        # Initialize global browser if needed
        if (_global_browser is None) or (cdp_url and cdp_url != "" and cdp_url != None):
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )
            
        if _global_browser_context is None or (chrome_cdp and cdp_url != "" and cdp_url != None):
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    no_viewport=False,
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )
            
        # Create a custom agent specifically for testing
        additional_info = f"Test Type: {test_type}. " + \
                        f"Test Scope: {scope}. " + \
                        f"{'Check links, ' if test_config.get('check_links', False) else ''}" + \
                        f"{'Check forms, ' if test_config.get('check_forms', False) else ''}" + \
                        f"{'Check navigation, ' if test_config.get('check_navigation', False) else ''}" + \
                        f"{'Check contrast, ' if test_config.get('check_contrast', False) else ''}" + \
                        f"{'Check alt text, ' if test_config.get('check_alt_text', False) else ''}" + \
                        f"{'Check ARIA, ' if test_config.get('check_aria', False) else ''}" + \
                        f"{'Check loading, ' if test_config.get('check_loading', False) else ''}" + \
                        f"{'Check resources, ' if test_config.get('check_resources', False) else ''}" + \
                        f"{'Check responsiveness' if test_config.get('check_responsiveness', False) else ''}"
        
        logger.info(f"Creating CustomAgent for testing with task: {task}")
        _global_agent = CustomAgent(
            task=task,
            add_infos=additional_info,
            use_vision=use_vision,
            llm=llm,
            browser=_global_browser,
            browser_context=_global_browser_context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            max_actions_per_step=15,  # Increased from 10 for more efficiency per step
            tool_calling_method="auto",
            generate_gif=True
        )
        
        # Run the agent with the specified number of steps
        logger.info(f"Running CustomAgent for website testing with max_steps={max_steps}")
        history = await _global_agent.run(max_steps=max_steps)
        
        # Save agent history
        history_file = os.path.join(save_agent_history_path, f"{test_type}_test_{timestamp}.json")
        _global_agent.save_history(history_file)
        
        # Extract results
        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()
        
        # Ensure results are always strings for Markdown components
        if not isinstance(final_result, str):
            final_result = str(final_result) if final_result is not None else "No results available."
            
        if not isinstance(errors, str):
            errors = str(errors) if errors is not None else "No errors occurred."
        
        # Generate a report file
        report_dir = os.path.join("tmp", "test_reports")
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"{test_type}_test_{timestamp}.md")
        with open(report_path, "w") as f:
            f.write(f"# {test_type.title()} Test Results for {target_url}\n\n")
            f.write(f"Test performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(final_result)
        
        logger.info(f"Website test complete, report saved to {report_path}")
        
        # Get the result GIF
        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")
        
        # Get the latest trace file
        trace_files = get_latest_files(save_trace_path)
        trace_file = trace_files.get('.zip')
        
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None
                
        _global_agent = None
        return final_result, errors, model_actions, model_thoughts, gif_path, trace_file, history_file, gr.update(interactive=True)
        
    except Exception as e:
        import traceback
        error_msg = f"Test error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Website test failed: {error_msg}")
        
        # Clean up in case of error
        _global_agent = None
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None
                
        return f"Error: {str(e)}", error_msg, "", "", None, None, None, gr.update(interactive=True)


# Helper function to run all steps of website testing
async def handle_website_test(test_type, test_url, 
                            func_check_links, func_check_forms, func_check_navigation, func_depth,
                            access_wcag_level, access_check_contrast, access_check_alt_text, access_check_aria,
                            perf_check_loading, perf_check_resources, perf_check_responsiveness, perf_device_types,
                            llm_provider, llm_model_name, llm_num_ctx, llm_temperature, 
                            llm_base_url, llm_api_key, use_vision, use_own_browser, 
                            headless, chrome_cdp, test_max_steps, test_scope):
    """Handle the complete website testing process including collecting config and running tests"""
    
    # First collect the test configuration based on test type
    if test_type == "functionality":
        test_config = {
            "check_links": func_check_links,
            "check_forms": func_check_forms,
            "check_navigation": func_check_navigation,
            "depth": func_depth,
            "scope": test_scope
        }
    elif test_type == "accessibility":
        test_config = {
            "wcag_level": access_wcag_level,
            "check_contrast": access_check_contrast,
            "check_alt_text": access_check_alt_text,
            "check_aria": access_check_aria,
            "scope": test_scope
        }
    elif test_type == "performance":
        test_config = {
            "check_loading": perf_check_loading,
            "check_resources": perf_check_resources,
            "check_responsiveness": perf_check_responsiveness,
            "device_types": perf_device_types,
            "scope": test_scope
        }
    else:
        test_config = {"scope": test_scope}
    
    # Use default values for the additional parameters
    disable_security = True
    window_w = 1280
    window_h = 1100
    keep_browser_open = False
    
    # Then run the website test with the collected config
    results = await run_website_test(test_type, test_url, test_config, 
                                llm_provider, llm_model_name, llm_num_ctx, llm_temperature,
                                llm_base_url, llm_api_key, use_vision, use_own_browser,
                                headless, disable_security, window_w, window_h, 
                                chrome_cdp, keep_browser_open, test_max_steps)
    
    # Unpack results to match output structure expected by UI
    summary, errors, model_actions, model_thoughts, gif_path, trace_file, history_file, button_update = results
    
    # Return values to match the UI outputs
    return summary, errors, history_file, gif_path, trace_file, history_file, button_update


def create_ui(theme_name="Ocean"):
    css = """
    :root {
        --primary-color: #4A6EE0;
        --primary-color-hover: #3955b5;
        --secondary-color: #6FA8DC;
        --text-color: #333333;
        --background-color: #f9f9fd;
        --card-background: #ffffff;
        --border-color: #e0e0e0;
        --shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        --border-radius: 10px;
    }

    body {
        background-color: var(--background-color);
    }

    .gradio-container {
        max-width: 1300px !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 30px !important;
    }

    .header-text {
        text-align: center;
        margin-bottom: 32px;
        padding: 25px 0;
        background: linear-gradient(135deg, #4A6EE0, #6FA8DC);
        color: white;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
    }

    .header-text h1 {
        color: white;
        margin-bottom: 0.5em;
        font-size: 2.5em;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }

    .header-text h3 {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        font-size: 1.2em;
        max-width: 800px;
        margin: 0 auto;
    }

    .card {
        background-color: var(--card-background);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    .tab-active {
        border-bottom: 3px solid var(--primary-color) !important;
        color: var(--primary-color) !important;
        font-weight: bold;
    }

    button.primary {
        background: linear-gradient(to right, var(--primary-color), var(--secondary-color)) !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(74, 110, 224, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    button.primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(74, 110, 224, 0.4) !important;
    }

    button.secondary {
        background-color: var(--secondary-color) !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    button.secondary:hover {
        transform: translateY(-2px) !important;
    }

    button.stop {
        background-color: #d32f2f !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }

    button.stop:hover {
        background-color: #b71c1c !important;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        color: #666;
        font-size: 0.9em;
        border-top: 1px solid var(--border-color);
    }

    /* Custom styling for specific components */
    #run-agent-section {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .action-buttons {
        display: flex;
        gap: 15px;
        margin: 20px 0;
    }

    .results-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }

    @media (max-width: 768px) {
        .results-section {
            grid-template-columns: 1fr;
        }
    }

    /* Improve form layout */
    label {
        font-weight: 500;
        color: var(--text-color);
        margin-bottom: 6px;
    }

    input, select, textarea {
        border-radius: var(--border-radius) !important;
        border: 1px solid var(--border-color) !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
    }
    
    /* Tooltip improvements */
    .gr-input-label span {
        opacity: 0.75;
        font-size: 0.9em;
        font-style: italic;
    }
    
    /* Better focus states */
    input:focus, select:focus, textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(74, 110, 224, 0.15) !important;
        outline: none !important;
    }
    
    /* Checkbox and radio styling */
    input[type="checkbox"], input[type="radio"] {
        accent-color: var(--primary-color);
    }
    
    /* Slider improvements */
    input[type="range"] {
        height: 5px;
        background-color: #e0e0e0;
        border-radius: 5px;
    }
    
    input[type="range"]::-webkit-slider-thumb {
        background-color: var(--primary-color);
        width: 18px;
        height: 18px;
        border-radius: 50%;
        cursor: pointer;
    }
    
    /* Improve visibility of text on video/gif components */
    .caption-text-overlay {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 8px 12px;
        border-radius: 5px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }
    
    /* Style for video and gif components */
    video::cue, .caption {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        text-shadow: 1px 1px 2px black;
    }
    
    /* Improve visibility of labels on dark backgrounds */
    .video-container, .gif-container {
        position: relative;
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .video-container:hover, .gif-container:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    .video-container .label, .gif-container .label {
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 8px;
        border-radius: 4px;
    }
    
    /* Section headers */
    .section-header {
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--secondary-color);
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Accordion improvements */
    .gradio-accordion {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
        margin-bottom: 16px;
    }
    
    .gradio-accordion summary {
        padding: 12px 16px;
        background-color: #f5f7fd;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    
    .gradio-accordion summary:hover {
        background-color: #e9edf9;
    }
    
    /* File upload area styling */
    .file-upload {
        border: 2px dashed var(--border-color);
        border-radius: var(--border-radius);
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .file-upload:hover {
        border-color: var(--primary-color);
        background-color: rgba(74, 110, 224, 0.05);
    }
    
    /* Waiting browser session container */
    .waiting-container {
        width: 100%;
        height: 50vh;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px dashed #ddd;
        border-radius: var(--border-radius);
        background: linear-gradient(to bottom right, #f9f9fd, #f0f3fa);
    }
    
    .waiting-container h2 {
        color: #888;
        font-weight: 400;
        text-align: center;
    }
    
    /* Agent Run Container styling */
    .agent-run-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        background-color: white;
        border-radius: var(--border-radius);
        padding: 20px;
        box-shadow: var(--shadow);
    }
    
    /* Task Input styling */
    .task-input textarea, .additional-info textarea {
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius) !important;
        transition: all 0.3s ease;
        font-size: 1.05em !important;
    }
    
    .task-input textarea:focus, .additional-info textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(74, 110, 224, 0.15) !important;
    }
    
    /* Run and Stop button styling */
    .run-button {
        font-size: 1.1em !important;
        padding: 10px 20px !important;
    }
    
    .stop-button {
        font-size: 1.1em !important;
    }
    
    /* Results tabs styling */
    .results-tabs {
        margin-top: 20px;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    .results-tab, .details-tab {
        padding: 15px !important;
    }
    
    /* Result and error box styling */
    .result-box textarea, .error-box textarea, 
    .action-box textarea, .thought-box textarea {
        font-family: monospace !important;
        line-height: 1.5 !important;
    }
    
    .result-box textarea {
        background-color: #f0f8ff !important;
        border-left: 4px solid var(--primary-color) !important;
    }
    
    .error-box textarea {
        background-color: #fff0f0 !important;
        border-left: 4px solid #d32f2f !important;
    }
    
    .action-box textarea {
        background-color: #f0fff0 !important;
        border-left: 4px solid #4caf50 !important;
    }
    
    .thought-box textarea {
        background-color: #fff8e1 !important;
        border-left: 4px solid #ff9800 !important;
    }
    
    /* Recordings row styling */
    .recordings-row {
        margin-top: 10px;
        gap: 20px;
    }
    
    /* File download styling */
    .file-download {
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 10px;
        transition: all 0.3s ease;
    }
    
    .file-download:hover {
        border-color: var(--primary-color);
        background-color: rgba(74, 110, 224, 0.05);
    }
    
    /* Browser view styling improvements */
    .browser-view {
        margin-top: 15px;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    /* Story input styling */
    .story-input textarea {
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius) !important;
        background-color: #fcfcff !important;
        transition: all 0.3s ease;
    }
    
    .story-input textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(74, 110, 224, 0.15) !important;
        background-color: white !important;
    }
    
    /* Story Agent Tab styling */
    .story-agent-tab {
        padding: 0 !important;
    }
    
    .story-config-column {
        padding: 20px !important;
        border-right: 1px solid var(--border-color);
    }
    
    .story-output-column {
        padding: 20px !important;
        background-color: rgba(249, 250, 255, 0.5);
    }
    
    /* Model settings accordion */
    .model-settings-accordion,
    .output-options-accordion,
    .local-generation-accordion {
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    /* Provider and model dropdowns */
    .provider-dropdown,
    .model-dropdown,
    .image-model-dropdown {
        margin-bottom: 10px;
    }
    
    /* Server URL input styling */
    .server-url-input input {
        font-family: monospace !important;
    }
    
    /* Steps and guidance sliders */
    .steps-slider, .guidance-slider {
        padding: 10px !important;
        background: rgba(242, 244, 255, 0.5) !important;
        border-radius: var(--border-radius) !important;
    }
    
    /* Negative prompt input */
    .negative-prompt-input textarea {
        font-style: italic !important;
        background-color: #fff6f6 !important;
        border-left: 3px solid #ffcccc !important;
    }
    
    /* Variable duration checkbox and input */
    .variable-duration-checkbox {
        margin-top: 10px !important;
    }
    
    .variable-durations-input input {
        font-family: monospace !important;
    }
    
    /* Story run and stop buttons */
    .story-run-button {
        background: linear-gradient(to right, #4A6EE0, #7986CB) !important;
        box-shadow: 0 4px 8px rgba(74, 110, 224, 0.3) !important;
        padding: 10px 20px !important;
        margin-top: 16px !important;
        font-size: 1.1em !important;
        transition: all 0.3s ease !important;
    }
    
    .story-run-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(74, 110, 224, 0.4) !important;
    }
    
    .story-stop-button {
        background-color: #d32f2f !important;
        color: white !important;
        margin-top: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .story-stop-button:hover {
        background-color: #b71c1c !important;
    }
    
    /* Story error output */
    .story-errors textarea {
        background-color: #fff0f0 !important;
        border-left: 4px solid #d32f2f !important;
        color: #d32f2f !important;
        font-family: monospace !important;
    }
    
    /* Story script output */
    .story-script textarea {
        background-color: #fff !important;
        border-left: 4px solid #4A6EE0 !important;
        font-family: 'Georgia', serif !important;
        line-height: 1.6 !important;
        padding: 20px !important;
    }
    
    /* Story output tabs */
    .story-output-tabs {
        margin-top: 20px;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    /* Previous stories accordion */
    .previous-stories-accordion {
        margin-top: 20px;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    /* Story list styling */
    .story-list {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
    }
    
    /* Refresh button */
    .refresh-button {
        margin: 10px !important;
    }
    """

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_map[theme_name], css=css
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance - Seamlessly automate web tasks with intelligent browser control
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings", id=1, elem_classes="card"):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value="custom",
                        info="Select the type of agent to use",
                        interactive=True
                    )
                    with gr.Row():
                        with gr.Column():
                            max_steps = gr.Slider(
                                minimum=1,
                                maximum=200,
                                value=100,
                                step=1,
                                label="Max Run Steps",
                                info="Maximum number of steps the agent will take",
                                interactive=True
                            )
                        with gr.Column():
                            max_actions_per_step = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=10,
                                step=1,
                                label="Max Actions per Step",
                                info="Maximum number of actions the agent will take per step",
                                interactive=True
                            )
                    with gr.Row():
                        with gr.Column():
                            use_vision = gr.Checkbox(
                                label="Use Vision",
                                value=True,
                                info="Enable visual processing capabilities",
                                interactive=True
                            )
                        with gr.Column():
                            max_input_tokens = gr.Number(
                                label="Max Input Tokens",
                                value=128000,
                                precision=0,
                                interactive=True
                            )
                            tool_calling_method = gr.Dropdown(
                                label="Tool Calling Method",
                                value="auto",
                                interactive=True,
                                allow_custom_value=True,  # Allow users to input custom model names
                                choices=["auto", "json_schema", "function_calling"],
                                info="Tool Calls Function Name",
                                visible=False
                            )

            with gr.TabItem("🔧 LLM Settings", id=2, elem_classes="card"):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            llm_provider = gr.Dropdown(
                                choices=[provider for provider, model in utils.model_names.items()],
                                label="LLM Provider",
                                value="ollama",
                                info="Select your preferred language model provider",
                                interactive=True
                            )
                        with gr.Column():
                            llm_model_name = gr.Dropdown(
                                label="Model Name",
                                choices=utils.model_names['ollama'],
                                value="deepseek-r1:14b",
                                interactive=True,
                                allow_custom_value=True,  # Allow users to input custom model names
                                info="Select a model or type a custom model name"
                            )
                    with gr.Row():
                        with gr.Column():
                            ollama_num_ctx = gr.Slider(
                                minimum=2 ** 8,
                                maximum=2 ** 16,
                                value=16000,
                                step=1,
                                label="Ollama Context Length",
                                info="Controls max context length (less = faster)",
                                visible=False,
                                interactive=True
                            )
                        with gr.Column():
                            llm_temperature = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.6,
                                step=0.1,
                                label="Temperature",
                                info="Controls randomness in model outputs",
                                interactive=True
                            )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value="",
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value="",
                            info="Your API key (leave blank to use .env)"
                        )

            # Change event to update context length slider
            def update_llm_num_ctx_visibility(llm_provider):
                return gr.update(visible=llm_provider == "ollama")

            # Bind the change event of llm_provider to update the visibility of context length slider
            llm_provider.change(
                fn=update_llm_num_ctx_visibility,
                inputs=llm_provider,
                outputs=ollama_num_ctx
            )

            with gr.TabItem("🌐 Browser Settings", id=3, elem_classes="card"):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            use_own_browser = gr.Checkbox(
                                label="Use Own Browser",
                                value=False,
                                info="Use your existing browser instance",
                                interactive=True
                            )
                            keep_browser_open = gr.Checkbox(
                                label="Keep Browser Open",
                                value=False,
                                info="Keep Browser Open between Tasks",
                                interactive=True
                            )
                        with gr.Column():
                            headless = gr.Checkbox(
                                label="Headless Mode",
                                value=False,
                                info="Run browser without GUI",
                                interactive=True
                            )
                            disable_security = gr.Checkbox(
                                label="Disable Security",
                                value=True,
                                info="Disable browser security features",
                                interactive=True
                            )
                            enable_recording = gr.Checkbox(
                                label="Enable Recording",
                                value=True,
                                info="Enable saving browser recordings",
                                interactive=True
                            )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=1280,
                            info="Browser window width",
                            interactive=True
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=1100,
                            info="Browser window height",
                            interactive=True
                        )

                    chrome_cdp = gr.Textbox(
                        label="CDP URL",
                        placeholder="http://localhost:9222",
                        value="",
                        info="CDP for Google remote debugging",
                        interactive=True
                    )

                    with gr.Accordion("Storage Paths", open=False):
                        save_recording_path = gr.Textbox(
                            label="Recording Path",
                            placeholder="e.g. ./tmp/record_videos",
                            value="./tmp/record_videos",
                            info="Path to save browser recordings",
                            interactive=True
                        )

                        save_trace_path = gr.Textbox(
                            label="Trace Path",
                            placeholder="e.g. ./tmp/traces",
                            value="./tmp/traces",
                            info="Path to save Agent traces",
                            interactive=True,
                        )

                        save_agent_history_path = gr.Textbox(
                            label="Agent History Save Path",
                            placeholder="e.g., ./tmp/agent_history",
                            value="./tmp/agent_history",
                            info="Directory where agent history should be saved",
                            interactive=True,
                        )

            with gr.TabItem("🤖 Run Agent", id=4, elem_classes="card"):
                with gr.Group(elem_id="run-agent-section", elem_classes=["agent-run-container"]):
                    gr.Markdown("### 📋 Task Definition", elem_classes=["section-header"])
                    task = gr.Textbox(
                        label="Task Description",
                        lines=4,
                        placeholder="Enter your task here...",
                        value="go to google.com and type 'OpenAI' click search and give me the first url",
                        info="Describe what you want the agent to do",
                        elem_classes=["task-input"]
                    )
                    add_infos = gr.Textbox(
                        label="Additional Information",
                        lines=3,
                        placeholder="Add any helpful context or instructions...",
                        info="Optional hints to help the LLM complete the task",
                        value="",
                        elem_classes=["additional-info"]
                    )

                    with gr.Row(elem_classes=["action-buttons"]):
                        run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2, elem_classes=["run-button"])
                        stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1, elem_classes=["stop-button"])

                    browser_view = gr.HTML(
                        value="<div class='waiting-container'><h2>Waiting for browser session...<br><small>Your automated browser will appear here</small></h2></div>",
                        label="Live Browser View",
                        visible=False,
                        elem_classes=["browser-view"]
                    )

                    with gr.Tabs(elem_classes=["results-tabs"]):
                        with gr.TabItem("Results Summary", elem_classes=["results-tab"]):
                            gr.Markdown("### 📊 Results", elem_classes=["section-header"])
                            with gr.Row():
                                with gr.Column():
                                    final_result_output = gr.Textbox(
                                        label="Final Result", 
                                        lines=5, 
                                        show_label=True,
                                        elem_classes=["result-box"]
                                    )
                                with gr.Column():
                                    errors_output = gr.Textbox(
                                        label="Errors", 
                                        lines=5, 
                                        show_label=True,
                                        elem_classes=["error-box"]
                                    )
                        
                        with gr.TabItem("Execution Details", elem_classes=["details-tab"]):
                            with gr.Row():
                                with gr.Column():
                                    model_actions_output = gr.Textbox(
                                        label="Model Actions", 
                                        lines=10, 
                                        show_label=True,
                                        elem_classes=["action-box"]
                                    )
                                with gr.Column():
                                    model_thoughts_output = gr.Textbox(
                                        label="Model Thoughts", 
                                        lines=10, 
                                        show_label=True,
                                        elem_classes=["thought-box"]
                                    )
                    
                    gr.Markdown("### 📹 Recordings & Files", elem_classes=["section-header"])
                    with gr.Row(elem_classes=["recordings-row"]):
                        with gr.Column():
                            recording_gif = gr.Image(
                                label="Result GIF", 
                                format="gif", 
                                elem_classes=["gif-container"]
                            )
                        with gr.Column():
                            with gr.Row():
                                trace_file = gr.File(
                                    label="Trace File",
                                    elem_classes=["file-download"]
                                )
                                agent_history_file = gr.File(
                                    label="Agent History",
                                    elem_classes=["file-download"]
                                )

            with gr.Tab("💭 Story Agent", elem_classes=["story-agent-tab"]):
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["story-config-column"]):
                        gr.Markdown("### 📝 Story Configuration", elem_classes=["section-header"])
                        story_task = gr.Textbox(
                            label="Story Topic", 
                            value="A space adventure with a brave astronaut and her robot companion exploring a new planet", 
                            lines=3,
                            placeholder="Describe the story you want to generate...",
                            elem_classes=["story-input"]
                        )
                        
                        with gr.Accordion("Model Settings", open=True, elem_classes=["model-settings-accordion"]):
                            with gr.Row():
                                story_llm_provider = gr.Dropdown(
                                    label="Language Model Provider",
                                    choices=["openai", "anthropic", "google", "azure", "ollama", "anyscale", "openrouter", "custom"],
                                    value="openrouter",
                                    elem_classes=["provider-dropdown"]
                                )
                                
                                story_llm_model_name = gr.Dropdown(
                                    label="Language Model",
                                    choices=llm_models.get("openrouter", []),
                                    value="qwen/qwen3-30b-a3b:free",
                                    elem_classes=["model-dropdown"]
                                )
                                
                            with gr.Row():
                                story_image_model = gr.Dropdown(
                                    label="Image Generation Model",
                                    choices=["gpt-image-1", "dall-e-3", "dall-e-2"],
                                    value="dall-e-3",
                                    elem_classes=["image-model-dropdown"]
                                )
                                
                                story_use_local_generation = gr.Checkbox(
                                    label="Use Local Image Generation",
                                    value=False,
                                    elem_classes=["local-generation-checkbox"]
                                )
                        
                        # Add local generation settings (hidden by default)
                        with gr.Accordion("Local Image Generation Settings", open=False, visible=False, elem_classes=["local-generation-accordion"]) as local_sd_accordion:
                            story_local_generation_url = gr.Textbox(
                                label="Server URL",
                                value="http://localhost:8000",
                                info="URL of your local Stable Diffusion server",
                                elem_classes=["server-url-input"]
                            )
                            
                            with gr.Row():
                                story_local_generation_steps = gr.Slider(
                                    minimum=1,
                                    maximum=150,
                                    value=20,
                                    step=1,
                                    label="Inference Steps",
                                    info="Higher values = better quality but slower generation",
                                    elem_classes=["steps-slider"]
                                )
                                
                                story_local_generation_guidance = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.5,
                                    step=0.5,
                                    label="Guidance Scale",
                                    info="How closely to follow prompt (higher = more faithful)",
                                    elem_classes=["guidance-slider"]
                                )
                                
                            story_local_generation_negative = gr.Textbox(
                                label="Negative Prompt",
                                value="low quality, bad anatomy, worst quality, low resolution",
                                info="Things to avoid in generated images",
                                elem_classes=["negative-prompt-input"]
                            )
                        
                        # Show/hide local SD settings based on checkbox
                        story_use_local_generation.change(
                            fn=lambda use_local: gr.update(visible=use_local),
                            inputs=[story_use_local_generation],
                            outputs=[local_sd_accordion]
                        )
                        
                        with gr.Accordion("Output Options", open=True, elem_classes=["output-options-accordion"]):
                            with gr.Row():
                                story_save_path = gr.Textbox(
                                    label="Save Path",
                                    placeholder="e.g. ./story_output",
                                    value="./story_output",
                                    elem_classes=["save-path-input"]
                                )
                                
                                story_use_seed = gr.Checkbox(
                                    label="Use Image Seed",
                                    value=True,
                                    elem_classes=["seed-checkbox"]
                                )
                            
                            with gr.Row():
                                story_gif_duration = gr.Number(
                                    label="GIF Frame Duration (s)",
                                    value=0.05,
                                    elem_classes=["duration-input"]
                                )
                                
                                story_video_duration = gr.Number(
                                    label="Video Frame Duration (s)",
                                    value=5.0,
                                    elem_classes=["duration-input"]
                                )
                                
                                story_video_framerate = gr.Number(
                                    label="Video Framerate",
                                    value=2,
                                    elem_classes=["framerate-input"]
                                )
                                
                            story_use_variable = gr.Checkbox(
                                label="Use Variable Durations",
                                value=False,
                                info="Enable to specify different durations for each frame",
                                elem_classes=["variable-duration-checkbox"]
                            )
                            
                            story_variable_durations = gr.Textbox(
                                label="Variable Durations (comma-separated seconds)",
                                value="3.0, 2.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0",
                                info="Specify the duration for each frame in seconds, comma-separated",
                                visible=False,
                                elem_classes=["variable-durations-input"]
                            )
                            
                            # Show/hide variable durations input based on checkbox
                            story_use_variable.change(
                                fn=lambda use_var: gr.update(visible=use_var),
                                inputs=[story_use_variable],
                                outputs=[story_variable_durations]
                            )
                        
                        story_run_button = gr.Button("▶️ Generate Story", variant="primary", elem_classes=["story-run-button"])
                        story_stop_button = gr.Button("⏹ Stop", variant="stop", elem_classes=["story-stop-button"])
                    
                    with gr.Column(scale=3, elem_classes=["story-output-column"]):
                        story_errors_output = gr.Textbox(
                            label="Errors",
                            value="",
                            visible=False,
                            lines=6,
                            elem_classes=["story-errors"]
                        )
                        
                        with gr.Tabs(elem_classes=["story-output-tabs"]):
                            with gr.TabItem("Story Script", elem_classes=["script-tab"]):
                                story_script_output = gr.Textbox(
                                    label="📚 Story Script",
                                    lines=15,
                                    elem_classes=["story-script"]
                                )
                            
                            with gr.TabItem("Animation", elem_classes=["animation-tab"]):
                                with gr.Tabs():
                                    with gr.TabItem("GIF"):
                                        story_image_output = gr.Image(
                                            label="🎬 Story Animation",
                                            type="filepath",
                                            height=600,
                                            elem_classes=["gif-container"]
                                        )
                                    with gr.TabItem("Video"):
                                        story_video_output = gr.Video(
                                            label="🎬 Story Video",
                                            height=600,
                                            elem_classes=["video-container"]
                                        )
                            
                            with gr.TabItem("Download", elem_classes=["download-tab"]):
                                story_files_output = gr.File(
                                    label="📦 Download Story Files",
                                    file_count="multiple",
                                    elem_classes=["story-files"]
                                )
                
                # Section for previously generated stories
                with gr.Accordion("📚 Previous Stories", open=False, elem_classes=["previous-stories-accordion"]):
                    with gr.Row():
                        story_refresh_button = gr.Button("🔄 Refresh", variant="secondary", scale=1, elem_classes=["refresh-button"])
                    
                    story_list = gr.Markdown("Click Refresh to see previously generated stories", elem_classes=["story-list"])

                # Define the list_story_folders function here
                def list_story_folders(base_path):
                    """List the timestamped story folders with their creation dates"""
                    if not base_path or not os.path.exists(base_path):
                        # Return a message instead of an error when the directory doesn't exist
                        return "No stories found. Base directory doesn't exist."
                    
                    try:
                        # Get all subdirectories in the base path
                        story_dirs = [d for d in os.listdir(base_path) 
                                    if os.path.isdir(os.path.join(base_path, d))]
                        
                        if not story_dirs:
                            return "No stories found in the base directory."
                        
                        # Sort directories by creation time (newest first)
                        story_dirs.sort(key=lambda d: os.path.getctime(os.path.join(base_path, d)), reverse=True)
                        
                        # Format the output as markdown
                        result = "## Previous Stories\n\n"
                        result += "| Date | Story | Files |\n"
                        result += "|------|-------|-------|\n"
                        
                        for d in story_dirs:
                            # Get creation time
                            ctime = os.path.getctime(os.path.join(base_path, d))
                            from datetime import datetime
                            date_str = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Check for story files
                            dir_path = os.path.join(base_path, d)
                            gif_path = os.path.join(dir_path, "story.gif")
                            video_path = os.path.join(dir_path, "story.mp4")
                            script_path = os.path.join(dir_path, "story_script.json")
                            
                            gif_exists = "✅" if os.path.exists(gif_path) else "❌"
                            video_exists = "✅" if os.path.exists(video_path) else "❌"
                            script_exists = "✅" if os.path.exists(script_path) else "❌"
                            
                            # Format directory name
                            dir_name = d
                            if "_" in d and d[0].isdigit():  # If it's a timestamped name
                                # Try to extract a more readable name after the timestamp
                                parts = d.split("_", 2)  # Split at most 2 times
                                if len(parts) > 2:
                                    dir_name = parts[2].replace("_", " ")
                            
                            result += f"| {date_str} | {dir_name} | GIF: {gif_exists} Video: {video_exists} Script: {script_exists} |\n"
                        
                        return result
                    except Exception as e:
                        # Handle any other errors gracefully
                        import traceback
                        logger.error(f"Error in list_story_folders: {e}\n{traceback.format_exc()}")
                        return "Error listing story folders. Check logs for details."
                
                # Event handlers for the story agent
                def update_story_model_choices(provider):
                    return gr.update(choices=llm_models.get(provider, []))
                
                story_llm_provider.change(
                    fn=update_story_model_choices,
                    inputs=[story_llm_provider],
                    outputs=[story_llm_model_name]
                )
                
                story_refresh_button.click(
                    fn=list_story_folders,
                    inputs=[story_save_path],
                    outputs=[story_list]
                )

            with gr.TabItem("🧐 Deep Research", id=5, elem_classes="card"):
                research_task_input = gr.Textbox(
                    label="Research Task", 
                    lines=5,
                    value="Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects, substantiated with examples of relevant models and techniques. The report should reflect original insights and analysis, moving beyond mere summarization of existing literature.",
                    interactive=True
                )
                with gr.Row():
                    with gr.Column():
                        max_search_iteration_input = gr.Number(
                            label="Max Search Iteration", 
                            value=3,
                            precision=0,
                            interactive=True
                        )
                    with gr.Column():
                        max_query_per_iter_input = gr.Number(
                            label="Max Query per Iteration", 
                            value=1,
                            precision=0,
                            interactive=True
                        )
                with gr.Row(elem_classes="action-buttons"):
                    research_button = gr.Button("▶️ Run Deep Research", variant="primary", scale=2)
                    stop_research_button = gr.Button("⏹ Stop", variant="stop", scale=1)
                
                with gr.Accordion("Research Results", open=True):
                    markdown_output_display = gr.Markdown(label="Research Report")
                    markdown_download = gr.File(label="Download Research Report")

             # Add the deepsite tab after all LLM components are defined
            with gr.TabItem("🌐 DeepSite Generator", elem_classes="card"):
                with gr.Group():
                    deepsite_description = gr.Markdown("""
                    # DeepSite Website Generator
                    
                    Generate complete websites using HTML, CSS, and JavaScript. The generator uses TailwindCSS by default for styling.
                    
                    Simply describe what kind of website you want, and the AI will create it for you.
                    
                    Note: This feature uses the LLM settings configured in the "LLM Settings" tab.
                    """)
                    
                    deepsite_input = gr.Textbox(
                        label="Website Description",
                        placeholder="Describe the website you want to create...",
                        lines=5,
                        info="Be as detailed as possible about the website's purpose, content, style, and functionality"
                    )
                    
                    with gr.Row():
                        deepsite_generate_btn = gr.Button("Generate Website", variant="primary")
                        deepsite_clear_btn = gr.Button("Clear", variant="secondary")
                    
                    deepsite_output = gr.HTML(
                        label="Generated Website"
                    )
                    
                    deepsite_code = gr.Code(
                        label="HTML Code",
                        language="html",
                        visible=False
                    )
                    
                    deepsite_view_code_btn = gr.Button("View Code")
                    deepsite_download_btn = gr.Button("Download HTML File")
                    deepsite_file_path = gr.Textbox(visible=False)
                    
                    # Define DeepSite functions - now using global LLM settings
                    async def generate_website(description, provider, model_name, ctx, temperature, base_url, api_key):
                        try:
                            print(f"Generating website with provider: {provider}, model: {model_name}")
                            
                            # Get the LLM model using the global LLM settings
                            current_llm = utils.get_llm_model(
                                provider=provider,
                                model_name=model_name,
                                num_ctx=ctx if provider == "ollama" else 8000,
                                temperature=temperature,
                                base_url=base_url,
                                api_key=api_key,
                            )
                            
                            print("LLM model created successfully")
                            
                            # Run the deepsite agent
                            html_content, errors, _, _, _, file_path = await run_deepsite_agent(
                                llm=current_llm,
                                task=description,
                                max_steps=10,
                                use_vision=True
                            )
                            
                            if errors:
                                print(f"DeepSite agent returned errors: {errors}")
                                return gr.update(value=f"<p>Error generating website: {errors}</p>"), "", gr.update(visible=False), ""
                            
                            print(f"DeepSite generated HTML successfully, saved to {file_path}")
                            return gr.update(value=html_content), html_content, gr.update(visible=True), file_path
                        except Exception as e:
                            import traceback
                            error_details = str(e) + "\n" + traceback.format_exc()
                            print(f"Error in generate_website: {error_details}")
                            return gr.update(value=f"<p>Error generating website: {str(e)}</p>"), "", gr.update(visible=False), ""
                    
                    def clear_deepsite():
                        return "", "", gr.update(visible=False), ""
                    
                    def toggle_code_view(code):
                        return gr.update(visible=True), code
                    
                    def download_html(file_path):
                        if file_path:
                            return gr.update(value=file_path)
                        return gr.update(value="")
                    
                    # Connect DeepSite UI components - now using global LLM settings
                    deepsite_generate_btn.click(
                        fn=generate_website,
                        inputs=[
                            deepsite_input,
                            llm_provider,
                            llm_model_name,
                            ollama_num_ctx,
                            llm_temperature,
                            llm_base_url,
                            llm_api_key
                        ],
                        outputs=[deepsite_output, deepsite_code, deepsite_code, deepsite_file_path]
                    )
                    
                    deepsite_clear_btn.click(
                        clear_deepsite,
                        inputs=[],
                        outputs=[deepsite_input, deepsite_output, deepsite_code, deepsite_file_path]
                    )
                    
                    deepsite_view_code_btn.click(
                        toggle_code_view,
                        inputs=[deepsite_code],
                        outputs=[deepsite_code, deepsite_code]
                    )
                    
                    deepsite_download_btn.click(
                        download_html,
                        inputs=[deepsite_file_path],
                        outputs=[gr.File(label="Download", type="filepath")]
                    )

            with gr.TabItem("🔬 Website Testing", id=6, elem_classes="card"):
                with gr.Group():
                    test_url = gr.Textbox(
                        label="Website URL",
                        placeholder="https://example.com",
                        value="",
                        info="URL of the website to test",
                        interactive=True
                    )
                    
                    test_type = gr.Radio(
                        ["functionality", "accessibility", "performance"],
                        label="Test Type",
                        value="functionality",
                        info="Select what aspect of the website to test",
                        interactive=True
                    )
                    
                    # Test execution settings
                    with gr.Row():
                        with gr.Column():
                            test_max_steps = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=25,
                                step=5,
                                label="Maximum Test Steps",
                                info="More steps allow for deeper testing but take longer",
                                interactive=True
                            )
                        with gr.Column():
                            test_scope = gr.Radio(
                                ["homepage only", "limited navigation", "thorough exploration"],
                                label="Test Scope",
                                value="limited navigation",
                                info="Controls how deeply the agent explores the website",
                                interactive=True
                            )
                    
                    # Dynamic configuration based on test type
                    with gr.Row(visible=True) as functionality_config:
                        with gr.Column():
                            func_check_links = gr.Checkbox(label="Check Links", value=True, 
                                info="Test if links on the page are valid")
                            func_check_forms = gr.Checkbox(label="Check Forms", value=True,
                                info="Test form submissions")
                        with gr.Column():
                            func_check_navigation = gr.Checkbox(label="Check Navigation", value=True,
                                info="Test navigation flows")
                            func_depth = gr.Slider(label="Test Depth", minimum=1, maximum=3, value=1, step=1,
                                info="How many levels of pages to test")
                    
                    with gr.Row(visible=False) as accessibility_config:
                        with gr.Column():
                            access_wcag_level = gr.Dropdown(
                                ["A", "AA", "AAA"],
                                label="WCAG Compliance Level",
                                value="AA",
                                info="Web Content Accessibility Guidelines level to test against"
                            )
                            access_check_contrast = gr.Checkbox(label="Check Color Contrast", value=True,
                                info="Test color contrast ratios")
                        with gr.Column():
                            access_check_alt_text = gr.Checkbox(label="Check Alt Text", value=True,
                                info="Test for image alt text")
                            access_check_aria = gr.Checkbox(label="Check ARIA", value=True,
                                info="Test ARIA attributes")
                    
                    with gr.Row(visible=False) as performance_config:
                        with gr.Column():
                            perf_check_loading = gr.Checkbox(label="Page Load Time", value=True,
                                info="Measure page load time")
                            perf_check_resources = gr.Checkbox(label="Resource Usage", value=True,
                                info="Analyze resource usage")
                        with gr.Column():
                            perf_check_responsiveness = gr.Checkbox(label="Responsiveness", value=True,
                                info="Test responsiveness on different screen sizes")
                            perf_device_types = gr.CheckboxGroup(
                                ["Desktop", "Tablet", "Mobile"],
                                label="Device Types",
                                value=["Desktop"],
                                info="Device types to test responsiveness on"
                            )
                    
                    # Run test button
                    with gr.Row(elem_classes="action-buttons"):
                        run_test_button = gr.Button("▶️ Run Test", variant="primary")
                        
                    # Results displays
                    with gr.Accordion("Test Results", open=True):
                        test_summary = gr.Markdown(label="Summary")
                        test_details = gr.Markdown(label="Details")
                        
                    with gr.Row():
                        with gr.Column():
                            test_report_output = gr.Markdown(value="Test Results will appear here")
                        with gr.Column():
                            test_recording_gif = gr.Image(label="Test Recording", format="gif", elem_classes=["gif-container"])
                    
                    with gr.Row():
                        with gr.Column():
                            test_trace_file = gr.File(label="Trace File")
                        with gr.Column():
                            test_agent_history = gr.File(label="Agent History")
                    
                    test_report = gr.File(label="Download Full Report")
                    
                    # Function to update visibility of config sections based on test type
                    def update_test_config_visibility(test_type):
                        return {
                            functionality_config: test_type == "functionality",
                            accessibility_config: test_type == "accessibility",
                            performance_config: test_type == "performance"
                        }
                    
                    # Connect the test type selection to update the config visibility
                    test_type.change(
                        fn=update_test_config_visibility,
                        inputs=test_type,
                        outputs=[functionality_config, accessibility_config, performance_config]
                    )
                    
                    # Connect run test button
                    run_test_button.click(
                        fn=handle_website_test,
                        inputs=[
                            test_type, test_url,
                            func_check_links, func_check_forms, func_check_navigation, func_depth,
                            access_wcag_level, access_check_contrast, access_check_alt_text, access_check_aria,
                            perf_check_loading, perf_check_resources, perf_check_responsiveness, perf_device_types,
                            llm_provider, llm_model_name, ollama_num_ctx, llm_temperature, 
                            llm_base_url, llm_api_key, use_vision, use_own_browser, 
                            headless, chrome_cdp, test_max_steps, test_scope
                        ],
                        outputs=[
                            test_summary, 
                            test_details, 
                            test_report,
                            test_recording_gif, 
                            test_trace_file, 
                            test_agent_history, 
                            run_test_button
                        ]
                    )

            with gr.TabItem("🎥 Recordings", id=7, elem_classes="card"):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []

                    # Get all video files
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(
                        os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

                    # Sort recordings by creation time (oldest first)
                    recordings.sort(key=os.path.getctime)

                    # Add numbering to the recordings
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))

                    return numbered_recordings

                with gr.Row():
                    refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                
                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    columns=3,
                    height="auto",
                    object_fit="contain",
                    elem_classes="card"
                )
                
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )

            with gr.TabItem("📁 UI Configuration", id=8, elem_classes="card"):
                config_file_input = gr.File(
                    label="Load UI Settings from Config File",
                    file_types=[".json"],
                    interactive=True
                )
                with gr.Row(elem_classes="action-buttons"):
                    load_config_button = gr.Button("Load Config", variant="primary")
                    save_config_button = gr.Button("Save UI Settings", variant="primary")

                config_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
                save_config_button.click(
                    fn=save_current_config,
                    inputs=[],
                    outputs=[config_status]
                )
                
        # Add a footer with additional information
        with gr.Row(elem_classes=["footer"]):
            gr.Markdown(
                """
                **Browser Use WebUI** - AI-powered web automation tool
                
                Made with ❤️ by the Browser Use team
                """
            )

        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model_name
        )

        # Add this after defining the components
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

        scan_and_register_components(demo)
        global webui_config_manager
        all_components = webui_config_manager.get_all_components()

        load_config_button.click(
            fn=update_ui_from_config,
            inputs=[config_file_input],
            outputs=all_components + [config_status]
        )

        # Bind the stop button click event after errors_output is defined
        stop_button.click(
            fn=stop_agent,
            inputs=[],
            outputs=[stop_button, run_button],
        )

        # Run button click handler
        run_button.click(
            fn=run_with_stream,
            inputs=[
                agent_type, llm_provider, llm_model_name, ollama_num_ctx, llm_temperature, llm_base_url,
                llm_api_key,
                use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                save_recording_path, save_agent_history_path, save_trace_path,
                enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step,
                tool_calling_method, chrome_cdp, max_input_tokens
            ],
            outputs=[
                browser_view,
                final_result_output,
                errors_output,
                model_actions_output,
                model_thoughts_output,
                recording_gif,
                trace_file,
                agent_history_file,
                stop_button,
                run_button
            ],
        )

        # Run Deep Research
        research_button.click(
            fn=run_deep_search,
            inputs=[research_task_input, max_search_iteration_input, max_query_per_iter_input, llm_provider,
                    llm_model_name, ollama_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision,
                    use_own_browser, headless, chrome_cdp],
            outputs=[markdown_output_display, markdown_download, stop_research_button, research_button]
        )
        
        # Bind the stop button click event for research
        stop_research_button.click(
            fn=stop_research_agent,
            inputs=[],
            outputs=[stop_research_button, research_button],
        )

        # Story Agent click handlers
        async def on_story_run_click(
            story_task, 
            story_llm_provider,
            story_llm_model_name, 
            story_image_model,
            story_use_local_generation,
            story_local_generation_url,
            story_local_generation_steps,
            story_local_generation_guidance,
            story_local_generation_negative,
            story_save_path,
            story_use_seed,
            story_gif_duration,
            story_video_duration,
            story_video_framerate,
            story_use_variable, 
            story_variable_durations
        ):
            # Show a loading state
            yield gr.update(visible=False), "Generating story...", None, None, None, list_story_folders(story_save_path)
            
            try:
                # Configure LLM
                llm = utils.create_llm_from_params(
                    llm_provider=story_llm_provider,
                    llm_model_name=story_llm_model_name,
                    llm_temperature=0.7,
                    llm_num_ctx=4096,
                    llm_base_url="",
                    llm_api_key="",
                    max_input_tokens=8000
                )
                
                # Parse variable durations if enabled
                variable_durations_list = None
                if story_use_variable and story_variable_durations:
                    try:
                        # Parse comma-separated string into list of floats
                        variable_durations_list = [float(d.strip()) for d in story_variable_durations.split(',') if d.strip()]
                        logger.info(f"Using variable durations: {variable_durations_list}")
                    except Exception as e:
                        logger.error(f"Error parsing variable durations: {e}")
                
                # Add additional OpenAI API key if available
                image_generation_api_key = os.getenv("OPENAI_API_KEY", "")
                
                # Run the story agent
                from src.agent.story_agent import StoryAgent
                
                # Create the story agent with appropriate parameters
                story_agent = StoryAgent(
                    task=story_task,
                    llm=llm,
                    image_generation_model=story_image_model,
                    image_generation_api_key=image_generation_api_key,
                    save_story_path=story_save_path,
                    use_image_seed=story_use_seed,
                    gif_frame_duration=story_gif_duration,
                    video_frame_duration=story_video_duration,
                    video_framerate=story_video_framerate,
                    variable_durations=variable_durations_list,
                    # Local Stable Diffusion options
                    use_local_generation=story_use_local_generation,
                    local_generation_url=story_local_generation_url,
                    local_generation_steps=story_local_generation_steps,
                    local_generation_guidance_scale=story_local_generation_guidance,
                    local_generation_negative_prompt=story_local_generation_negative
                )
                
                # Set the global agent reference
                global _global_agent
                _global_agent = story_agent
                
                # Run the agent
                result = await story_agent.run()
                
                if not result.get("success", False):
                    errors = result.get("error", "Story generation failed")
                    yield gr.update(value=errors, visible=True), "Error occurred during story generation", None, None, None, list_story_folders(story_save_path)
                else:
                    # Get paths from result
                    gif_path = result.get("gif_path")
                    video_path = result.get("video_path") 
                    script_path = result.get("script_path")
                    
                    # Read the script content
                    script_content = ""
                    if script_path and os.path.exists(script_path):
                        try:
                            with open(script_path, 'r', encoding='utf-8') as f:
                                script_content = f.read()
                        except Exception as e:
                            logger.error(f"Error reading script file: {e}")
                    
                    # Get the actual folder path from the GIF path
                    story_folder = os.path.dirname(gif_path) if gif_path else ""
                    
                    # Add image generation method to output
                    image_method = result.get("image_generation_method", "unknown")
                    header = f"Story saved in: {story_folder}\n\n"
                    header += f"Image generation: {image_method}\n\n"
                    
                    formatted_script = header + (script_content or "Story generated successfully")
                    
                    # Create a list of files to download
                    download_files = []
                    if gif_path and os.path.exists(gif_path):
                        download_files.append(gif_path)
                    if video_path and os.path.exists(video_path):
                        download_files.append(video_path)
                    if script_path and os.path.exists(script_path):
                        download_files.append(script_path)
                    
                    # Update UI with results
                    video_output = video_path if video_path and os.path.exists(video_path) else None
                        
                    yield (
                        gr.update(visible=False), 
                        formatted_script, 
                        gif_path, 
                        video_output, 
                        download_files if download_files else None, 
                        list_story_folders(story_save_path)
                    )
                        
            except Exception as e:
                import traceback
                error_details = str(e) + "\n" + traceback.format_exc()
                yield gr.update(value=error_details, visible=True), "Error occurred during story generation", None, None, None, list_story_folders(story_save_path)

        story_run_button.click(
            fn=on_story_run_click,
            inputs=[
                story_task, 
                story_llm_provider, 
                story_llm_model_name, 
                story_image_model, 
                story_use_local_generation,
                story_local_generation_url,
                story_local_generation_steps,
                story_local_generation_guidance,
                story_local_generation_negative,
                story_save_path, 
                story_use_seed,
                story_gif_duration,
                story_video_duration,
                story_video_framerate,
                story_use_variable,
                story_variable_durations
            ],
            outputs=[
                story_errors_output, 
                story_script_output, 
                story_image_output, 
                story_video_output, 
                story_files_output, 
                story_list
            ]
        )

        async def on_story_stop_click():
            global _global_agent
            if _global_agent is not None:
                _global_agent.stop()
            return "Story generation stopped"
            
        story_stop_button.click(
            fn=on_story_stop_click,
            outputs=[story_script_output]
        )
        
        return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    demo = create_ui(theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()
