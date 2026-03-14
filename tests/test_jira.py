import httpx
import pytest
import respx

from app.config import JiraConfig
from app.jira import JiraClient, JiraTicket, _extract_acceptance_criteria, _strip_jira_markup


@pytest.fixture
def jira_config():
    return JiraConfig(base_url="https://jira.example.com", token="test-token")


@pytest.fixture
def jira_client(jira_config):
    return JiraClient(jira_config)


def _jira_response(
    key="SEP-22888",
    summary="Config security for intranet",
    description="Implement security config",
    labels=None,
    subtasks=None,
    issue_type=None,
    status=None,
):
    fields = {
        "summary": summary,
        "description": description,
        "labels": labels or [],
        "subtasks": subtasks or [],
    }
    if issue_type is not None:
        fields["issuetype"] = {"name": issue_type}
    if status is not None:
        fields["status"] = {"name": status}
    return {"key": key, "fields": fields}


class TestFetchTicket:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_success(self, jira_client):
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-22888").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(
                    labels=["security", "backend"],
                    subtasks=[
                        {"key": "SEP-22889", "fields": {"summary": "Implement auth filter"}},
                        {"key": "SEP-22890", "fields": {"summary": "Add config endpoint"}},
                    ],
                    issue_type="Story",
                    status="In Progress",
                ),
            )
        )

        ticket = await jira_client.fetch_ticket("SEP-22888")

        assert ticket is not None
        assert ticket.key == "SEP-22888"
        assert ticket.title == "Config security for intranet"
        assert ticket.description == "Implement security config"
        assert ticket.labels == ["security", "backend"]
        assert len(ticket.subtasks) == 2
        assert ticket.subtasks[0] == "SEP-22889: Implement auth filter"
        assert ticket.url == "https://jira.example.com/browse/SEP-22888"
        assert ticket.issue_type == "Story"
        assert ticket.status == "In Progress"

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_not_found(self, jira_client):
        respx.get("https://jira.example.com/rest/api/2/issue/NOPE-999").mock(
            return_value=httpx.Response(404)
        )

        ticket = await jira_client.fetch_ticket("NOPE-999")
        assert ticket is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_connection_error(self, jira_client):
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-123").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        ticket = await jira_client.fetch_ticket("SEP-123")
        assert ticket is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_truncates_description(self, jira_client):
        long_desc = "x" * 6000
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-1").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(key="SEP-1", description=long_desc),
            )
        )

        ticket = await jira_client.fetch_ticket("SEP-1")

        assert ticket is not None
        assert len(ticket.description) == JiraTicket.MAX_DESCRIPTION_LENGTH + 3  # +3 for "..."
        assert ticket.description.endswith("...")

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_no_description(self, jira_client):
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-2").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(key="SEP-2", description=None),
            )
        )

        ticket = await jira_client.fetch_ticket("SEP-2")

        assert ticket is not None
        assert ticket.description is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_type_and_status(self, jira_client):
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-4").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(key="SEP-4", issue_type="Bug", status="Done"),
            )
        )

        ticket = await jira_client.fetch_ticket("SEP-4")

        assert ticket is not None
        assert ticket.issue_type == "Bug"
        assert ticket.status == "Done"

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_no_type_or_status(self, jira_client):
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-5").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(key="SEP-5"),
            )
        )

        ticket = await jira_client.fetch_ticket("SEP-5")

        assert ticket is not None
        assert ticket.issue_type is None
        assert ticket.status is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_strips_markup_from_description(self, jira_client):
        desc = "h1. Overview\n*bold text* and _italic text_"
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-6").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(key="SEP-6", description=desc),
            )
        )

        ticket = await jira_client.fetch_ticket("SEP-6")

        assert ticket is not None
        assert "h1." not in ticket.description
        assert "Overview" in ticket.description
        assert "*" not in ticket.description
        assert "bold text" in ticket.description

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_extracts_acceptance_criteria(self, jira_client):
        desc = "Some description.\nAK-1: Must handle auth\nAK-2: Must log errors\nOther text."
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-7").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(key="SEP-7", description=desc),
            )
        )

        ticket = await jira_client.fetch_ticket("SEP-7")

        assert ticket is not None
        assert ticket.acceptance_criteria is not None
        assert "AK-1: Must handle auth" in ticket.acceptance_criteria
        assert "AK-2: Must log errors" in ticket.acceptance_criteria

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_ticket_no_ac_when_prefixes_empty(self):
        config = JiraConfig(
            base_url="https://jira.example.com", token="test-token",
            acceptance_criteria_prefixes=[],
        )
        client = JiraClient(config)
        desc = "AK-1: Must handle auth\nAK-2: Must log errors"
        respx.get("https://jira.example.com/rest/api/2/issue/SEP-8").mock(
            return_value=httpx.Response(
                200,
                json=_jira_response(key="SEP-8", description=desc),
            )
        )

        ticket = await client.fetch_ticket("SEP-8")

        assert ticket is not None
        assert ticket.acceptance_criteria is None


class TestStripJiraMarkup:
    def test_headings(self):
        assert _strip_jira_markup("h1. Title\nh2. Subtitle") == "Title\nSubtitle"

    def test_code_blocks(self):
        result = _strip_jira_markup("{code:java}\nSystem.out.println();\n{code}")
        assert "System.out.println();" in result
        assert "{code" not in result

    def test_noformat(self):
        result = _strip_jira_markup("{noformat}\nsome text\n{noformat}")
        assert "some text" in result
        assert "{noformat}" not in result

    def test_color_tags(self):
        assert _strip_jira_markup("{color:red}warning{color}") == "warning"

    def test_images_removed(self):
        assert _strip_jira_markup("text !image.png! more") == "text  more"

    def test_links(self):
        assert _strip_jira_markup("[Google|https://google.com]") == "Google (https://google.com)"

    def test_table_headers(self):
        result = _strip_jira_markup("||Name||Age||")
        assert " | " in result
        assert "||" not in result

    def test_bold_italic_strikethrough(self):
        assert _strip_jira_markup("*bold* _italic_ -strike-") == "bold italic strike"

    def test_panel(self):
        result = _strip_jira_markup("{panel:title=Info}\ncontent here\n{panel}")
        assert "content here" in result
        assert "{panel" not in result

    def test_quote(self):
        result = _strip_jira_markup("{quote}\nquoted text\n{quote}")
        assert "quoted text" in result
        assert "{quote}" not in result

    def test_collapses_blank_lines(self):
        result = _strip_jira_markup("line1\n\n\n\n\nline2")
        assert result == "line1\n\nline2"

    def test_combined_markup(self):
        text = "h1. Overview\n{noformat}\ncode here\n{noformat}\n*important* [link|http://x.com]"
        result = _strip_jira_markup(text)
        assert "Overview" in result
        assert "code here" in result
        assert "important" in result
        assert "link (http://x.com)" in result
        assert "h1." not in result
        assert "{noformat}" not in result
        assert "*" not in result

    def test_image_with_attributes(self):
        assert _strip_jira_markup("!screenshot.png|width=500!") == ""


class TestExtractAcceptanceCriteria:
    def test_basic_ak_prefixes(self):
        desc = "Some text\nAK-1: Must handle auth\nAK-2: Must log errors\nOther"
        result = _extract_acceptance_criteria(desc, ["AK"])
        assert result is not None
        assert "AK-1: Must handle auth" in result
        assert "AK-2: Must log errors" in result

    def test_akzeptanzkriterium_prefix(self):
        desc = "Akzeptanzkriterium 1: Login works\nAkzeptanzkriterium 2: Logout works"
        result = _extract_acceptance_criteria(desc, ["Akzeptanzkriterium"])
        assert result is not None
        assert "Login works" in result
        assert "Logout works" in result

    def test_case_insensitive(self):
        desc = "ak-1: lowercase\nAK-2: uppercase"
        result = _extract_acceptance_criteria(desc, ["AK"])
        assert result is not None
        assert "lowercase" in result
        assert "uppercase" in result

    def test_no_matches(self):
        desc = "No acceptance criteria here."
        result = _extract_acceptance_criteria(desc, ["AK", "Akzeptanzkriterium"])
        assert result is None

    def test_empty_prefixes(self):
        desc = "AK-1: Something"
        result = _extract_acceptance_criteria(desc, [])
        assert result is None

    def test_multiple_prefixes(self):
        desc = "AK-1: First\nAC-1: Second"
        result = _extract_acceptance_criteria(desc, ["AK", "AC"])
        assert result is not None
        assert "AK-1: First" in result
        assert "AC-1: Second" in result

    def test_no_duplicates(self):
        desc = "AK-1: Same line"
        result = _extract_acceptance_criteria(desc, ["AK", "AK"])
        assert result is not None
        assert result.count("AK-1: Same line") == 1

    def test_various_separators(self):
        desc = "AK 1: Space separator\nAK-2. Dot separator\nAK3 No separator"
        result = _extract_acceptance_criteria(desc, ["AK"])
        assert result is not None
        assert "AK 1: Space separator" in result
        assert "AK-2. Dot separator" in result
        assert "AK3 No separator" in result

    def test_english_ac_prefix(self):
        desc = "Overview\nAC-1: User can log in\nAC-2: User can log out\nNotes"
        result = _extract_acceptance_criteria(desc, ["AC"])
        assert result is not None
        assert "AC-1: User can log in" in result
        assert "AC-2: User can log out" in result

    def test_dod_prefix(self):
        desc = "DoD-1: All tests pass\nDoD-2: Code reviewed"
        result = _extract_acceptance_criteria(desc, ["DoD"])
        assert result is not None
        assert "DoD-1: All tests pass" in result
        assert "DoD-2: Code reviewed" in result

    def test_req_prefix(self):
        desc = "Req-1: Must support OAuth\nReq-2: Must encrypt data"
        result = _extract_acceptance_criteria(desc, ["Req"])
        assert result is not None
        assert "Req-1: Must support OAuth" in result
        assert "Req-2: Must encrypt data" in result

    def test_acceptance_criteria_full_prefix(self):
        desc = "Acceptance Criteria 1: Login works\nAcceptance Criterion 2: Logout works"
        result = _extract_acceptance_criteria(desc, ["Acceptance Criteria", "Acceptance Criterion"])
        assert result is not None
        assert "Login works" in result
        assert "Logout works" in result

    def test_prefix_without_number(self):
        desc = "DoD: All tests must pass\nDoD: Code reviewed"
        result = _extract_acceptance_criteria(desc, ["DoD"])
        assert result is not None
        assert "DoD: All tests must pass" in result
        assert "DoD: Code reviewed" in result

    def test_prefix_only_matches_at_line_start(self):
        desc = "This is not an AK-1 criterion\nAK-2: Real criterion"
        result = _extract_acceptance_criteria(desc, ["AK"])
        assert result is not None
        assert "This is not an AK-1 criterion" not in result
        assert "AK-2: Real criterion" in result
