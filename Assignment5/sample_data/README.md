# Sample test data for the Multimodal RAG production assignment

This folder contains a minimal but realistic dataset to demonstrate the production design and test the product capabilities described in the assignment.

## Contents

- `raw_documents/`: sample source files used for retrieval testing
- `metadata/`: tenant, workspace, document, conversation, feedback, connector, usage, and audit JSON seed data
- `generate_sample_data.py`: script to recreate the full sample dataset

## What this sample demonstrates

This dataset covers the same progression as the assignment roadmap:

1. Phase 1: baseline single-app retrieval
2. Phase 2: multi-document and workspace access
3. Phase 3: asynchronous ingestion and processing states
4. Phase 4: tenant isolation and RBAC
5. Phase 5: hybrid retrieval with dense + BM25 + RRF
6. Phase 6: image and table-aware retrieval
7. Phase 7: evaluation and observability
8. Phase 8: public API and shareable chat workflows
9. Phase 9: connectors, quotas, billing, and governance

## How to regenerate

```bash
cd "Assignments/Assignment5_Multimodal_RAG/sample_data"
python generate_sample_data.py
```

## Suggested testing scenarios

- Search for premium response SLA and confirm the correct document is retrieved.
- Compare access between workspaces and tenants to validate isolation.
- Review shareable chat and feedback samples for product workflows.
- Validate billing and quota summaries from `usage_events.json`.
- Check connector metadata for Google Drive, SharePoint, and S3 integrations.
- Review `audit_events.json` to test compliance and governance examples.

## Sample documents included

- `doc_alpha_policy.pdf`: support policy
- `doc_beta_finance.pdf`: finance and billing overview
- `doc_gamma_security.pdf`: security and access controls
- `doc_delta_knowledge.txt`: knowledge base summary

These files are intentionally small and fully synthetic so they can be used for local testing without external cloud services.
