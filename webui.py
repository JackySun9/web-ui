import pdb
import logging

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
        --primary-color: #4a6ee0;
        --secondary-color: #45a9b3;
        --background-color: #f9fafb;
        --card-background: #ffffff;
        --text-color: #333333;
        --border-color: #e0e0e0;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --border-radius: 8px;
    }

    body {
        background-color: var(--background-color);
    }

    .gradio-container {
        max-width: 1200px !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 20px !important;
    }

    .header-text {
        text-align: center;
        margin-bottom: 20px;
    }

    .header-text h1 {
        color: var(--primary-color);
        margin-bottom: 0.5em;
    }

    .header-text h3 {
        color: var(--secondary-color);
        font-weight: normal;
    }

    .card {
        background-color: var(--card-background);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }

    .tab-active {
        border-bottom: 2px solid var(--primary-color) !important;
        color: var(--primary-color) !important;
        font-weight: bold;
    }

    button.primary {
        background-color: var(--primary-color) !important;
        border: none !important;
    }

    button.secondary {
        background-color: var(--secondary-color) !important;
        border: none !important;
    }

    .footer {
        text-align: center;
        margin-top: 30px;
        color: #888;
        font-size: 0.9em;
    }

    /* Custom styling for specific components */
    #run-agent-section {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    .action-buttons {
        display: flex;
        gap: 10px;
        margin: 15px 0;
    }

    .results-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
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
    }

    input, select, textarea {
        border-radius: var(--border-radius) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Tooltip improvements */
    .gr-input-label span {
        opacity: 0.7;
        font-size: 0.9em;
    }
    
    /* Better focus states */
    input:focus, select:focus, textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(74, 110, 224, 0.2) !important;
    }
    """

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_map[theme_name], css=css
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
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
                with gr.Group(elem_id="run-agent-section"):
                    task = gr.Textbox(
                        label="Task Description",
                        lines=4,
                        placeholder="Enter your task here...",
                        value="go to google.com and type 'OpenAI' click search and give me the first url",
                        info="Describe what you want the agent to do",
                        interactive=True
                    )
                    add_infos = gr.Textbox(
                        label="Additional Information",
                        lines=3,
                        placeholder="Add any helpful context or instructions...",
                        info="Optional hints to help the LLM complete the task",
                        value="",
                        interactive=True
                    )

                    with gr.Row(elem_classes="action-buttons"):
                        run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                        stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

                    browser_view = gr.HTML(
                        value="<div style='width:100%; height:50vh; display:flex; align-items:center; justify-content:center; border:1px dashed #ccc; border-radius:8px;'><h2 style='color:#888;'>Waiting for browser session...</h2></div>",
                        label="Live Browser View",
                        visible=False
                    )

                    gr.Markdown("### Results", elem_classes="section-header")
                    with gr.Accordion("Results", open=True, elem_classes="results-section"):
                        with gr.Row():
                            with gr.Column():
                                final_result_output = gr.Textbox(
                                    label="Final Result", lines=3, show_label=True
                                )
                            with gr.Column():
                                errors_output = gr.Textbox(
                                    label="Errors", lines=3, show_label=True
                                )
                        with gr.Row():
                            with gr.Column():
                                model_actions_output = gr.Textbox(
                                    label="Model Actions", lines=3, show_label=True, visible=False
                                )
                            with gr.Column():
                                model_thoughts_output = gr.Textbox(
                                    label="Model Thoughts", lines=3, show_label=True, visible=False
                                )
                    
                    with gr.Row():
                        with gr.Column():
                            recording_gif = gr.Image(label="Result GIF", format="gif")
                        with gr.Column():
                            with gr.Row():
                                trace_file = gr.File(label="Trace File")
                                agent_history_file = gr.File(label="Agent History")

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
                            test_recording_gif = gr.Image(label="Test Recording", format="gif")
                        with gr.Column():
                            with gr.Row():
                                test_trace_file = gr.File(label="Trace File")
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
                
        # Add a footer
        gr.HTML(
            """<div class="footer">Browser Use WebUI - AI-powered browser automation - <a href="https://github.com/xianjianlf2/browser-use" target="_blank">View on GitHub</a></div>""",
            elem_classes=["footer"]
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
