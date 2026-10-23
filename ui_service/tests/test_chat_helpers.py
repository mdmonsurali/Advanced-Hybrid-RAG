"""Chat UI helpers: settings -> pipeline options, citations, answer footer, escaping."""

from __future__ import annotations

from types import SimpleNamespace

import main
import pytest
from chat import citations, pipeline_stream, settings


def test_drive_command_only_for_allowed_roles():
    assert "drive" not in [c["id"] for c in settings.commands("user")]
    assert [c["id"] for c in settings.commands("admin")] == ["docs", "web", "drive", "catalog"]


def test_switched_off_sources_become_denied_routes():
    opts = settings.pipeline_options({"web": False, "answer_style": "concise"}, command="drive")
    assert opts == {
        "route_hint": ["gdrive"],
        "deny_routes": ["web"],
        "answer_style": "concise",
        "debug": False,
    }
    assert settings.pipeline_options(None, None)["deny_routes"] == []


@pytest.mark.parametrize(
    ("link", "safe"),
    [
        ("https://example.com/a", True),
        ("/documents/d1/source", True),
        ("javascript:alert(1)", False),
        ("//evil.example/x", False),
        ("data:text/html,<script>", False),
    ],
)
def test_only_http_and_same_site_links_are_shown(link, safe):
    assert (citations._link({"link": link}) is not None) is safe


def test_source_elements_are_named_like_the_citation_markers(monkeypatch):
    # cl.Text needs a live Chainlit session; a plain record of its arguments is enough here.
    monkeypatch.setattr(citations.cl, "Text", lambda **kw: SimpleNamespace(**kw))
    [el] = citations.source_elements(
        [{"n": 2, "route": "docs", "source": "policy.pdf", "page": 4, "snippet": "Line one",
          "link": "javascript:alert(1)"}]
    )  # fmt: skip
    assert el.name == "[2]"
    assert "**policy.pdf · p. 4**" in el.content
    assert "> Line one" in el.content
    assert "Open source" not in el.content  # the unsafe link was dropped


def test_footer_summarises_the_answer():
    footer = pipeline_stream._footer(
        {
            "action": "accept",
            "verdict": {"grounded_score": 0.87, "pii_masked": 1},
            "routes": ["docs", "web"],
            "attempts": 2,
            "elapsed_sec": 4.2,
        }
    )
    assert "groundedness 87%" in footer
    assert "docs + web" in footer
    assert "1 PII redacted" in footer
    assert "2 attempts" in footer
    assert footer.endswith("4.2s")


def test_html_escaping():
    assert (
        main._e("<a href=\"x\">'&'</a>") == "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;"
    )
