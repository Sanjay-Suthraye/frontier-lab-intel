> ## Documentation Index
> Fetch the complete documentation index at: https://platform.kimi.ai/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Platform Changelog

> Review historical Kimi Open Platform feature releases, model launches, product improvements, and issue fixes.

This page is updated periodically with Kimi Open Platform product updates and related documentation changes.

**October 2026**
### 📦 Batch API now supports Kimi K3.1 Pro and Kimi K3

The Batch API now supports the `kimi-k3.1-pro` and `kimi-k3` models (previously limited to `kimi-k2.7-code` and `kimi-k2.6`), available on both the CN and Global platforms. Batch inference is still priced at 60% of standard rates; batch jobs on `kimi-k3.1-pro` and `kimi-k3` are also charged Cache Write fees, and because batch requests are processed asynchronously and concurrently, cache hits are provided on a best-effort basis. See the [Batch API Guide](https://platform.kimi.ai/docs/guide/use-batch-api.md) and [BatchJob Pricing](https://platform.kimi.ai/docs/pricing/batch.md).

**September 2026**

### 🔍 Web Search API

Launch web searches through `/v1/tools/search` and `/v1/tools/search_pro` and get structured results — ideal for Agent applications that orchestrate their own search logic. Organization- and project-level QPS limits are supported. See [Web Search Basic](https://platform.kimi.ai/docs/api/tools-search.md), [Web Search Pro](https://platform.kimi.ai/docs/api/tools-search-pro.md), [Web Search Best Practices](https://platform.kimi.ai/docs/guide/best-practices-for-web-search.md), and [Web Search Pricing](https://platform.kimi.ai/docs/pricing/websearch.md).

### 💰 Billing method change

Account usage is now deducted at a uniform ratio of 50% cash and 50% vouchers. When either balance is depleted, all subsequent usage is deducted from the remaining balance type. Users with signed contracts follow their contract terms and continue to prioritize voucher usage — they are not affected.

**July 2026**
