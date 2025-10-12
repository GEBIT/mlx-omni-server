import re
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from ....utils.logger import logger


class ThinkingDecoder(ABC):
    """Base class for thinking decoders."""

    @abstractmethod
    def _parse_stream_response(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        pass

    @abstractmethod
    def _parse_response(self, response: str) -> Optional[Dict[str, Optional[str]]]:
        pass

    def stream_decode(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        """Parse tool calls from model output."""
        return self._parse_stream_response(text)

    def decode(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        """Parse thinking content from model output"""
        return self._parse_response(text)


class DefaultThinkingDecoder(ThinkingDecoder):
    """Default implementation of the thinking decoder."""

    def __init__(self, thinking_tag: str = "think", init_buffer: str = ""):
        self.thinking_tag = thinking_tag
        self.accumulated_text = init_buffer

    def _parse_stream_response(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        # Check if in thinking mode
        thinking_end_tag = f"</{self.thinking_tag}>"
        thinking_start_tag = f"<{self.thinking_tag}>"

        # Special case: text exactly equals the start tag
        if text == thinking_start_tag:
            self.accumulated_text += text
            return {"delta_content": None, "delta_thinking": ""}

        # Special case: text exactly equals the end tag
        if text == thinking_end_tag:
            self.accumulated_text += text
            return {
                "delta_content": "",
                "delta_thinking": None,
            }

        # Update accumulated text
        self.accumulated_text += text

        # Check if accumulated text already contains the end tag
        has_end_tag_before = thinking_end_tag in self.accumulated_text[: -len(text)]
        has_end_tag_now = thinking_end_tag in self.accumulated_text

        # If text starts with thinking tag and end tag hasn't been encountered yet
        if self.accumulated_text.startswith(thinking_start_tag) and not has_end_tag_now:
            # delta content as thinking
            return {"delta_content": None, "delta_thinking": text}
        # If current delta contains the end tag (from not having it to having it)
        elif not has_end_tag_before and has_end_tag_now:
            # Split the current delta
            parts = text.split(thinking_end_tag, 1)
            return {"delta_content": "", "delta_thinking": None}
        # If end tag was already encountered before
        elif has_end_tag_before:
            # All content after the end tag is content
            return {"delta_content": text, "delta_thinking": None}
        else:
            # Other cases, possibly thinking mode not enabled or other situations
            return {"delta_content": text, "delta_thinking": None}

    def _parse_response(self, response: str):
        tag = self.thinking_tag
        # First check for complete thinking tag pattern
        thinking_regex = rf"<{tag}>([\s\S]*?)</{tag}>"
        thinking_match = re.search(thinking_regex, response)

        if thinking_match:
            # Extract thinking content
            thinking_content = thinking_match.group(1).strip()

            # Get final content by replacing thinking tag and its content
            content = re.sub(thinking_regex, "", response, count=1).strip()

            return {
                "content": content,
                "thinking": thinking_content,
            }
        else:
            # Check if only end tag exists (missing start tag case)
            end_tag = f"</{tag}>"
            if end_tag in response:
                # Split response using end tag
                parts = response.split(end_tag, 1)
                if len(parts) == 2:
                    thinking_content = parts[0].strip()
                    content = parts[1].strip()
                    # If thinking_content is empty, treat as no thinking
                    if not thinking_content:
                        thinking_content = None
                    return {
                        "content": content,
                        "thinking": thinking_content,
                    }

            # If no tags exist, the entire response is the content
            return {
                "content": response.strip(),
                "thinking": None,
            }


class GptOssThinkingDecoder(ThinkingDecoder):
    """Thinking decoder for GPT-OSS model with custom tags."""

    def __init__(self):
        self.last_tag: str = "<|start|>"
        self.role: str = "assistant"
        self.channel: str = ""
        logger.info("Initialized GptOssThinkingDecoder")

    def _parse_stream_response(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        delta_content = ""
        delta_thinking = ""
        while len(text) > 0:
            if self.last_tag == "<|start|>":
                channel_idx = text.find("<|channel|>")
                if channel_idx >= 0:
                    self.role += text[:channel_idx]
                    self.last_tag = "<|channel|>"
                    self.channel = ""
                    text = text[channel_idx + len("<|channel|>") :]
                else:
                    self.role += text
                    text = ""
            elif self.last_tag == "<|channel|>":
                message_idx = text.find("<|message|>", 0)
                if message_idx >= 0:
                    self.channel += text[:message_idx]
                    self.last_tag = "<|message|>"
                    text = text[message_idx + len("<|message|>") :]
                else:
                    self.channel += text
                    text = ""
            elif self.last_tag == "<|message|>":
                end_idx = text.find("<|end|>", 0)
                delta_text = ""
                if end_idx >= 0:
                    delta_text = text[:end_idx]
                    self.last_tag = "<|end|>"
                    text = text[end_idx + len("<|end|>") :]
                else:
                    delta_text = text
                    text = ""
                if self.channel == "analysis":
                    delta_thinking += delta_text
                else:
                    delta_content += delta_text
            elif self.last_tag == "<|end|>":
                start_idx = text.find("<|start|>", 0)
                if start_idx >= 0:
                    if start_idx > 0:
                        print(
                            f"WARN: Unexpected text between <|end|> and <|start|>: {text[:start_idx]}"
                        )
                    self.last_tag = "<|start|>"
                    text = text[start_idx + len("<|start|>") :]
                else:
                    print(
                        f"WARN: Unexpected text after <|end|> before <|start|>: {text}"
                    )
                    text = ""

        return {
            "delta_content": delta_content,
            "delta_thinking": delta_thinking,
        }

    def _parse_response(self, response: str) -> Optional[Dict[str, Optional[str]]]:
        result = self._parse_stream_response(response)
        if result is None:
            return None
        return {
            "content": result.get("delta_content"),
            "thinking": result.get("delta_thinking"),
        }
