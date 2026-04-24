from pydantic import BaseModel


class PullRequestProject(BaseModel):
    key: str


class PullRequestRepository(BaseModel):
    slug: str
    project: PullRequestProject


class PullRequestRef(BaseModel):
    id: str
    displayId: str
    latestCommit: str | None = None
    repository: PullRequestRepository | None = None


class PullRequestUser(BaseModel):
    name: str
    slug: str | None = None
    displayName: str | None = None


class PullRequestParticipant(BaseModel):
    user: PullRequestUser


class PullRequest(BaseModel):
    id: int
    title: str
    state: str | None = None
    fromRef: PullRequestRef
    toRef: PullRequestRef
    author: PullRequestParticipant
    createdDate: int | None = None  # Bitbucket epoch milliseconds


class CommentParent(BaseModel):
    id: int


class Comment(BaseModel):
    id: int
    text: str
    author: PullRequestUser
    parent: CommentParent | None = None


class WebhookPayload(BaseModel):
    eventKey: str
    pullRequest: PullRequest
    comment: Comment | None = None
    commentParentId: int | None = None
    actor: PullRequestUser | None = None


class ReviewFinding(BaseModel):
    file: str
    line: int
    severity: str  # "critical", "important"
    confidence: int | None = None
    comment: str
    suggestion: str | None = None
