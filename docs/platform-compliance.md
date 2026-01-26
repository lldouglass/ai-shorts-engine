# Platform Compliance Guide

This document outlines the Terms of Service compliance requirements for each supported platform in the AI Shorts Engine.

## YouTube

### API Usage
- **API**: YouTube Data API v3 and YouTube Analytics API
- **Auth**: OAuth 2.0 with Device Flow or Browser Redirect
- **Quota**: 10,000 units/day default (request increase for production)

### Content Requirements
- **Max Duration**: 60 seconds for Shorts
- **Aspect Ratio**: 9:16 (vertical)
- **Resolution**: 1920x1080 recommended
- **Format**: MP4 (H.264 video, AAC audio)

### Rate Limits
- **Uploads**: 50 videos/day default limit (configurable)
- **API Calls**: Subject to quota units

### ToS Compliance
- Content must comply with [YouTube Community Guidelines](https://www.youtube.com/howyoutubeworks/policies/community-guidelines/)
- AI-generated content must comply with YouTube's [AI content policies](https://support.google.com/youtube/answer/13742253)
- Do not use misleading metadata or spam techniques
- Respect copyright for all media assets

---

## Instagram

### API Usage
- **API**: Instagram Graph API (via Meta Graph API)
- **Auth**: Facebook Login OAuth 2.0
- **Account Type**: Requires Instagram Business or Creator account

### Content Requirements
- **Max Duration**: 60 seconds for Reels
- **Aspect Ratio**: 9:16 (vertical)
- **Resolution**: 1080x1920 recommended
- **Format**: MP4

### Rate Limits
- **Uploads**: 25 posts/day per account
- **API Calls**: 200 calls/user/hour

### Publishing Flow
1. Upload video to container
2. Poll container status until FINISHED
3. Publish container to account

### ToS Compliance
- Content must comply with [Instagram Community Guidelines](https://help.instagram.com/477434105621119)
- AI-generated content should be labeled appropriately
- Do not use automated tools for engagement manipulation
- Respect intellectual property rights

---

## TikTok

### API Usage
- **API**: TikTok Content Posting API v2
- **Auth**: TikTok Login Kit OAuth 2.0
- **Capability**: Requires Direct Post approval for auto-publishing

### Content Requirements
- **Max Duration**: 60 seconds (up to 10 minutes with approval)
- **Aspect Ratio**: 9:16 (vertical)
- **Resolution**: 1080x1920 recommended
- **Format**: MP4

### Rate Limits
- **Uploads**: 50 videos/day per account
- **File Size**: 4GB max

### Publishing Modes

#### Direct Post (requires approval)
- Full automated publishing
- Content posted immediately or scheduled
- Requires TikTok developer app approval

#### Inbox Upload (fallback)
- Video sent to TikTok app inbox
- User manually posts from app
- Used when Direct Post not available
- Returns `NEEDS_MANUAL_PUBLISH` status

### ToS Compliance
- Content must comply with [TikTok Community Guidelines](https://www.tiktok.com/community-guidelines)
- AI-generated content should follow [TikTok's synthetic media policy](https://www.tiktok.com/creators/creator-portal/en-us/community-guidelines-and-safety/ai-generated-content/)
- Do not use deceptive practices for engagement
- Respect music and copyright licensing

---

## Common Compliance Requirements

### AI-Generated Content Disclosure
All platforms increasingly require disclosure of AI-generated content:

1. **YouTube**: Use content labels for synthetic media
2. **Instagram**: Label AI content in descriptions
3. **TikTok**: Use AI content disclosure features

### Copyright Compliance
- Use licensed or royalty-free music
- Ensure all visual assets are properly licensed
- Respect intellectual property in generated content

### Data Privacy
- Handle user tokens securely (encrypted at rest)
- Implement token refresh before expiry
- Delete tokens when accounts are disconnected

### Rate Limit Handling
- Implement exponential backoff for rate limits
- Track daily upload counts per account
- Reset counters at UTC midnight

---

## Error Handling

### Retryable Errors
- Network timeouts (retry with backoff)
- Rate limit errors (wait and retry)
- 5xx server errors (retry with backoff)

### Non-Retryable Errors
- Invalid credentials (mark account revoked)
- Content policy violations (fail immediately)
- Missing permissions (fail immediately)

### Manual Intervention Required
- `NEEDS_MANUAL_PUBLISH`: Direct Post unavailable
- `AWAITING_APPROVAL`: Content under platform review

---

## Security Considerations

### Token Storage
- All OAuth tokens encrypted using Fernet symmetric encryption
- Encryption key from `SECRET_KEY` environment variable
- Tokens stored in PostgreSQL with encrypted columns

### Token Refresh
- YouTube: Device flow tokens last ~6 months, refresh proactively
- Instagram: Long-lived tokens last 60 days, refresh at 7 days before expiry
- TikTok: Access tokens last 24 hours, refresh tokens last 365 days

### Account Revocation
Accounts are automatically marked as revoked when:
- Token refresh fails with "expired" or "invalid" errors
- API returns unauthorized/forbidden responses
- User manually revokes access on platform

---

## Monitoring

### Metrics to Track
- Upload success/failure rates per platform
- API quota usage
- Token refresh success rates
- Rate limit hits

### Alerts
- Token expiry warnings (7 days before)
- Account revocation events
- Quota approaching limits
- Publishing failures
