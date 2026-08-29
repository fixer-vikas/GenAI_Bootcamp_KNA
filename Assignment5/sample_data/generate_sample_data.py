from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_documents"
META = ROOT / "metadata"


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_pdf(path: Path, title: str, body: str) -> None:
    text = f"BT /F1 18 Tf 50 740 Td ({title}) Tj 0 -24 Td /F1 11 Tf ({body}) Tj ET"
    stream = text.encode("latin-1", errors="replace")
    content = (
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n" + stream.decode("latin-1") + "\nendstream\nendobj\n"
    )
    objects = content.split("\nendobj\n")
    pdf = "%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf.encode("latin-1")))
        if i == len(objects):
            pdf += f"{i} 0 obj\n{obj}"
        else:
            pdf += f"{i} 0 obj\n{obj}endobj\n"
    xref_start = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    path.write_bytes(pdf.encode("latin-1", errors="replace"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_tests() -> None:
    documents = json.loads((META / "documents.json").read_text(encoding="utf-8"))["documents"]
    tenants = json.loads((META / "tenants.json").read_text(encoding="utf-8"))["tenants"]
    workspaces = json.loads((META / "workspaces.json").read_text(encoding="utf-8"))["workspaces"]
    feedback = json.loads((META / "feedback.json").read_text(encoding="utf-8"))["feedback"]
    connectors = json.loads((META / "connectors.json").read_text(encoding="utf-8"))["connectors"]
    audit = json.loads((META / "audit_events.json").read_text(encoding="utf-8"))["audit_events"]

    assert_condition(len(documents) >= 4, "At least four sample documents should exist")
    assert_condition(len(tenants) >= 2, "At least two tenants should exist")
    assert_condition(len(workspaces) >= 3, "At least three workspaces should exist")
    assert_condition(len(feedback) >= 2, "Feedback data should exist")
    assert_condition(any(c["type"] in {"google_drive", "sharepoint", "s3"} for c in connectors), "At least one external connector should exist")
    assert_condition(len(audit) >= 3, "Audit log events should exist")

    # Capability checks aligned with the assignment roadmap
    required_capabilities = {
        "workspaces": any("workspace" in doc["workspace_id"] for doc in documents),
        "multi_document": len(documents) > 1,
        "conversation_history": (META / "conversations.json").exists(),
        "shareable_chats": any(c.get("shareable") is True for c in json.loads((META / "conversations.json").read_text(encoding="utf-8"))["conversations"]),
        "user_feedback": len(feedback) > 0,
        "admin_analytics": (META / "usage_events.json").exists(),
        "connectors": len(connectors) > 0,
        "audit_logs": len(audit) > 0,
        "billing": any(t.get("billing_status") in {"active", "trial"} for t in tenants),
    }

    missing = [name for name, ok in required_capabilities.items() if not ok]
    assert_condition(not missing, f"Missing capabilities: {missing}")

    print("Sample data validation passed")
    print(f"Validated documents: {len(documents)}")
    print(f"Validated workspaces: {len(workspaces)}")
    print(f"Validated connectors: {len(connectors)}")
    print(f"Validated audit events: {len(audit)}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    pdf_docs = {
        "doc_alpha_policy.pdf": (
            "Customer Support Policy",
            "This policy defines case handling, service-level response windows, escalation, and account ownership for premium customers.",
        ),
        "doc_beta_finance.pdf": (
            "Finance and Billing Overview",
            "This document outlines monthly billing, invoice collection, usage quotas, subscription plans, and support exclusions.",
        ),
        "doc_gamma_security.pdf": (
            "Security and Access Controls",
            "This document covers tenant access, RBAC roles, audit logs, encryption, and legal retention requirements across workspaces.",
        ),
    }

    for name, (title, body) in pdf_docs.items():
        write_pdf(RAW / name, title, body)

    (RAW / "doc_delta_knowledge.txt").write_text(
        "Knowledge base summary: The company supports Google Drive, SharePoint, and S3 connectors. "
        "Customers may share chats, review history, and provide thumbs up or thumbs down feedback. "
        "Admins can monitor usage, quotas, and billing dashboards.",
        encoding="utf-8",
    )

    tenant_data = {
        "tenants": [
            {
                "tenant_id": "tenant-arka",
                "name": "Arka Labs",
                "plan": "enterprise",
                "quota_limit": 50000,
                "billing_status": "active",
                "retention_days": 365,
            },
            {
                "tenant_id": "tenant-zenith",
                "name": "Zenith Health",
                "plan": "pro",
                "quota_limit": 20000,
                "billing_status": "trial",
                "retention_days": 180,
            },
        ]
    }
    write_json(META / "tenants.json", tenant_data)

    workspace_data = {
        "workspaces": [
            {
                "workspace_id": "workspace-support",
                "tenant_id": "tenant-arka",
                "name": "Support Ops",
                "document_ids": ["doc-alpha-policy", "doc-delta-knowledge"],
                "owner_id": "user-001",
            },
            {
                "workspace_id": "workspace-finance",
                "tenant_id": "tenant-arka",
                "name": "Finance",
                "document_ids": ["doc-beta-finance"],
                "owner_id": "user-002",
            },
            {
                "workspace_id": "workspace-security",
                "tenant_id": "tenant-zenith",
                "name": "Security",
                "document_ids": ["doc-gamma-security"],
                "owner_id": "user-003",
            },
        ]
    }
    write_json(META / "workspaces.json", workspace_data)

    document_data = {
        "documents": [
            {
                "document_id": "doc-alpha-policy",
                "tenant_id": "tenant-arka",
                "workspace_id": "workspace-support",
                "name": "Customer Support Policy",
                "source_file": "raw_documents/doc_alpha_policy.pdf",
                "status": "ready",
                "owner_id": "user-001",
                "content_types": ["text", "table"],
                "pages": 2,
                "version": "v1",
            },
            {
                "document_id": "doc-beta-finance",
                "tenant_id": "tenant-arka",
                "workspace_id": "workspace-finance",
                "name": "Finance and Billing Overview",
                "source_file": "raw_documents/doc_beta_finance.pdf",
                "status": "ready",
                "owner_id": "user-002",
                "content_types": ["text", "table"],
                "pages": 2,
                "version": "v1",
            },
            {
                "document_id": "doc-gamma-security",
                "tenant_id": "tenant-zenith",
                "workspace_id": "workspace-security",
                "name": "Security and Access Controls",
                "source_file": "raw_documents/doc_gamma_security.pdf",
                "status": "ready",
                "owner_id": "user-003",
                "content_types": ["text", "image"],
                "pages": 2,
                "version": "v1",
            },
            {
                "document_id": "doc-delta-knowledge",
                "tenant_id": "tenant-arka",
                "workspace_id": "workspace-support",
                "name": "Knowledge Base Summary",
                "source_file": "raw_documents/doc_delta_knowledge.txt",
                "status": "ready",
                "owner_id": "user-001",
                "content_types": ["text"],
                "pages": 1,
                "version": "v1",
            },
        ]
    }
    write_json(META / "documents.json", document_data)

    conversation_data = {
        "conversations": [
            {
                "conversation_id": "conv-1001",
                "tenant_id": "tenant-arka",
                "workspace_id": "workspace-support",
                "user_id": "user-001",
                "messages": [
                    {"role": "user", "content": "What is our response SLA for premium customers?"},
                    {"role": "assistant", "content": "Premium customers should receive a response within 2 hours according to the Customer Support Policy."},
                ],
                "shareable": True,
            },
            {
                "conversation_id": "conv-1002",
                "tenant_id": "tenant-arka",
                "workspace_id": "workspace-finance",
                "user_id": "user-002",
                "messages": [
                    {"role": "user", "content": "What is the billing plan for enterprise accounts?"},
                    {"role": "assistant", "content": "Enterprise plans have active billing status and higher quota limits under the Finance and Billing Overview."},
                ],
                "shareable": False,
            },
        ]
    }
    write_json(META / "conversations.json", conversation_data)

    feedback_data = {
        "feedback": [
            {"feedback_id": "fb-1", "conversation_id": "conv-1001", "document_id": "doc-alpha-policy", "score": 1, "comment": "Helpful and grounded."},
            {"feedback_id": "fb-2", "conversation_id": "conv-1002", "document_id": "doc-beta-finance", "score": 0, "comment": "Missing pricing detail."},
        ]
    }
    write_json(META / "feedback.json", feedback_data)

    usage_data = {
        "usage_events": [
            {"event_id": "usage-1", "tenant_id": "tenant-arka", "workspace_id": "workspace-support", "user_id": "user-001", "tokens": 5600, "cost": 0.14, "feature": "chat", "time": "2026-08-01T10:00:00Z"},
            {"event_id": "usage-2", "tenant_id": "tenant-arka", "workspace_id": "workspace-finance", "user_id": "user-002", "tokens": 3200, "cost": 0.09, "feature": "retrieval", "time": "2026-08-01T10:10:00Z"},
            {"event_id": "usage-3", "tenant_id": "tenant-zenith", "workspace_id": "workspace-security", "user_id": "user-003", "tokens": 4100, "cost": 0.11, "feature": "chat", "time": "2026-08-01T10:20:00Z"},
        ],
        "quota_summary": {"tenant-arka": 50000, "tenant-zenith": 20000},
    }
    write_json(META / "usage_events.json", usage_data)

    connector_data = {
        "connectors": [
            {"connector_id": "conn-drive", "type": "google_drive", "tenant_id": "tenant-arka", "status": "connected", "credentials_ref": "secret://drive/arka"},
            {"connector_id": "conn-sharepoint", "type": "sharepoint", "tenant_id": "tenant-arka", "status": "connected", "credentials_ref": "secret://sharepoint/arka"},
            {"connector_id": "conn-s3", "type": "s3", "tenant_id": "tenant-zenith", "status": "connected", "credentials_ref": "secret://s3/zenith"},
        ]
    }
    write_json(META / "connectors.json", connector_data)

    audit_data = {
        "audit_events": [
            {"event_id": "audit-1", "tenant_id": "tenant-arka", "actor_id": "user-001", "action": "upload_document", "resource_id": "doc-alpha-policy", "result": "success"},
            {"event_id": "audit-2", "tenant_id": "tenant-arka", "actor_id": "user-002", "action": "share_chat", "resource_id": "conv-1002", "result": "success"},
            {"event_id": "audit-3", "tenant_id": "tenant-zenith", "actor_id": "user-003", "action": "view_audit_log", "resource_id": "workspace-security", "result": "success"},
        ]
    }
    write_json(META / "audit_events.json", audit_data)

    phase_data = {
        "phase_1": {"description": "Single-app retrieval baseline", "sample_documents": ["doc-alpha-policy"]},
        "phase_2": {"description": "Multiple documents and workspaces", "sample_documents": ["doc-alpha-policy", "doc-beta-finance", "doc-delta-knowledge"]},
        "phase_3": {"description": "Async ingestion and storage", "sample_jobs": [{"job_id": "job-101", "status": "ready"}]},
        "phase_4": {"description": "Tenant isolation and RBAC", "sample_roles": ["owner", "editor", "viewer"]},
        "phase_5": {"description": "Dense + BM25 + RRF retrieval", "sample_queries": ["What is the premium response SLA?", "How much is the supported billing quota?"]},
        "phase_6": {"description": "Image and table retrieval", "sample_documents": ["doc-beta-finance", "doc-gamma-security"]},
        "phase_7": {"description": "Evaluation and observability", "sample_metrics": ["latency", "groundedness", "feedback"]},
        "phase_8": {"description": "Public API and frontend demo", "sample_conversations": ["conv-1001", "conv-1002"]},
        "phase_9": {"description": "Connectors, quotas, billing, governance", "sample_connectors": ["conn-drive", "conn-sharepoint", "conn-s3"]},
    }
    write_json(META / "phase_plan.json", phase_data)

    print(f"Created sample dataset at {ROOT}")
    print(f"Generated PDFs: {', '.join(sorted(pdf_docs))}")
    print("Seed files:")
    for file in sorted(META.glob("*.json")):
        print(f" - {file.relative_to(ROOT)}")

    run_tests()


if __name__ == "__main__":
    main()
