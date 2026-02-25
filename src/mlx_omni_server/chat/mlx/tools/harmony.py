import re
import uuid
import json
from typing import List, Optional

from ....utils.logger import logger
from ..core_types import ToolCall
from .base_tools import BaseToolParser

# shoutout to https://github.com/ggml-org/llama.cpp/blob/92bb84f775215cf36e3224708a9a93e2531a0a96/common/chat.cpp#L1956
REGEX_PREFIX = r"(?:[\S\s]*)" # captures nothing
REGEX_CHANNEL = r"<\|channel\|>(?:analysis|commentary)" # captures nothing
REGEX_RECIPIENT = r"(?: to=functions\.([^<\s]+))" # captures function name
REGEX_CONSTRAINT = r"(?:\s*(?:<\|constrain\|>)?([a-zA-Z0-9_-]+))?" # captures constrain ("json")
REGEX_MESSAGE = r"(?:<\|message\|>)?([\S\s]*)" # captures body
REGEX_TOOL_CALL_1 = re.compile(REGEX_PREFIX + REGEX_RECIPIENT + REGEX_CHANNEL + REGEX_CONSTRAINT + REGEX_MESSAGE)
REGEX_TOOL_CALL_2 = re.compile(REGEX_PREFIX + REGEX_CHANNEL + REGEX_RECIPIENT + REGEX_CONSTRAINT + REGEX_MESSAGE)

class HarmonyToolParser(BaseToolParser):
    """Tools parser for the Harmony format used by GPT-OSS models."""

    def __init__(self):
        self.start_tool_calls = "" # forcing tool use is not supported as the model needs to think
        self.end_tool_calls = ""

    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output using simplified regex approach.

        Args:
            text: Generated text that may contain tool calls

        Returns:
            List of ToolCall objects or None if no tool calls found
        """
        if not text or not isinstance(text, str):
            return None

        try:
            match = REGEX_TOOL_CALL_1.match(text) or REGEX_TOOL_CALL_2.match(text)
            if match is None:
                return None
            else:
                groups = match.groups()
                function_name = groups[0].strip()
                if len(groups) == 3 and groups[1] is not None and groups[1].strip() != "json":
                    logger.warning(f"Unknown constrain: {groups[1]}")
                body = groups[-1].strip()
                return self._parse_call(function_name, body)

        except Exception as e:
            logger.error(f"Error parsing GPT-OSS tool calls: {e}")
            return None
    
    def _parse_call(self, function_name: str, args: str) -> Optional[List[ToolCall]]:
        arguments = {}
        try:
            if len(args.strip()) != 0:
                arguments = json.loads(args)
        except json.decoder.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON body: '{args}':", e)
        
        return [ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=function_name,
            arguments=arguments,
        )]
