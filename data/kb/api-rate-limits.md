# API rate limits
## Limits per plan
Plus allows 100 requests per minute and Pro allows 1000 requests per minute, counted across your whole account. The Free plan has no API access.
## Hitting the limit
When you exceed the limit the API returns HTTP 429. Read the Retry-After header and wait that many seconds before retrying. Batch requests where possible.
