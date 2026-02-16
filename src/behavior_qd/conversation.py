"""Data model for multi-turn conversations in behavior-qd."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """A multi-turn conversation thread.

    Stores the full conversation history and tracks which parent
    conversation (from the previous turn's archive) it extends.
    """

    turns: list[ConversationTurn] = field(default_factory=list)
    parent_id: int | None = None  # Entry ID in the previous turn's archive

    @property
    def last_user_message(self) -> str:
        """Get the last user message (the most recent continuation)."""
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return ""

    @property
    def last_assistant_message(self) -> str:
        """Get the last assistant message."""
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.content
        return ""

    @property
    def num_turns(self) -> int:
        """Number of complete user-assistant exchanges."""
        return sum(1 for t in self.turns if t.role == "user")

    @property
    def messages(self) -> list[dict[str, str]]:
        """Convert to API message format (list of role/content dicts)."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def with_continuation(
        self,
        user_message: str,
        assistant_response: str | None = None,
    ) -> Conversation:
        """Create a new conversation extending this one with additional turns.

        Args:
            user_message: The new user follow-up message.
            assistant_response: Optional assistant response to include.

        Returns:
            New Conversation with the appended turns.
        """
        new_turns = list(self.turns)
        new_turns.append(ConversationTurn(role="user", content=user_message))
        if assistant_response is not None:
            new_turns.append(
                ConversationTurn(role="assistant", content=assistant_response)
            )
        return Conversation(turns=new_turns, parent_id=self.parent_id)

    @classmethod
    def from_prompt_response(
        cls,
        prompt: str,
        response: str | None = None,
    ) -> Conversation:
        """Create a conversation from a single prompt-response pair.

        Args:
            prompt: The user's initial prompt.
            response: Optional assistant response.

        Returns:
            Conversation with one or two turns.
        """
        turns = [ConversationTurn(role="user", content=prompt)]
        if response is not None:
            turns.append(ConversationTurn(role="assistant", content=response))
        return cls(turns=turns)

    def format_for_display(self, max_content_length: int | None = None) -> str:
        """Format the conversation for human-readable display.

        Args:
            max_content_length: Truncate content to this length if set.

        Returns:
            Formatted string representation.
        """
        lines = []
        for turn in self.turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            content = turn.content
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            lines.append(f"[{role_label}]: {content}")
        return "\n".join(lines)
