from contextvars import ContextVar

import uuid
from fastapi import Depends, Request

# context_api_id stores unique id for every request
context_api_id: ContextVar[str] = ContextVar('api_id', default=None)
# context_log_meta stores log meta data for every request
context_log_meta: ContextVar[dict] = ContextVar('log_meta', default={})
