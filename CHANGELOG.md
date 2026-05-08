# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `backend/exceptions.py` with custom exception classes: `GeminiServiceError`, `ValidationError`, `RateLimitError`
- Global exception handlers for all three custom exception types plus a generic catch-all that prevents tracebacks reaching clients
- API versioning: all endpoints are now served under the `/api/v1/` prefix via an `APIRouter`
- Response models for every endpoint: `DestinationsResponse`, `SaveSearchResponse`, `RecentSearchesResponse`, `ConfigResponse`, `HealthResponse`, `PlacesResponse`, `SearchItem`
- OpenAPI `json_schema_extra` examples on all request and response schemas
- Full return-type annotations on all functions and endpoint handlers
- Docstrings with `Args`, `Returns`, and `Raises` sections on all public functions and helpers

### Changed
- `_PUBLIC_PATHS` updated from `/health` / `/config` to `/api/v1/health` / `/api/v1/config` to match new route prefixes
- Endpoint return values converted from plain dicts to typed response-model instances where applicable

## [1.0.0] - 2025-01-01

### Added
- Streaming Gemini chat endpoint with SSE
- Destination extraction from free-form text (Redis-cached 30 min)
- Google Places search proxy (Redis-cached 1 hour)
- Google Places photo proxy
- Firestore trip search history (save and retrieve)
- Upstash Redis caching layer with graceful fallback
- HTML minification at startup in production
- API-key authentication middleware (`X-API-Key` header)
- Request ID, security headers, and structured JSON access logging
- Google Cloud Logging integration with stdlib fallback
- Google Secret Manager integration with env-var fallback
