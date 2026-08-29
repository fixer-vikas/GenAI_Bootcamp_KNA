from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META = ROOT / "metadata"


def load_json(name: str):
    return json.loads((META / name).read_text(encoding="utf-8"))


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def demo_phase_1():
    print_section("Phase 1: baseline retrieval")
    docs = load_json("documents.json")["documents"]
    print("Available sample documents:")
    for doc in docs:
        print(f" - {doc['document_id']}: {doc['name']} ({doc['workspace_id']})")


def demo_phase_2():
    print_section("Phase 2: multi-document and workspaces")
    workspaces = load_json("workspaces.json")["workspaces"]
    for ws in workspaces:
        print(f"Workspace {ws['workspace_id']} -> documents: {', '.join(ws['document_ids'])}")


def demo_phase_3():
    print_section("Phase 3: async ingestion")
    jobs = [
        {"job_id": "job-101", "status": "uploaded"},
        {"job_id": "job-102", "status": "parsing"},
        {"job_id": "job-103", "status": "ready"},
    ]
    for job in jobs:
        print(f"Job {job['job_id']} is in state: {job['status']}")


def demo_phase_4():
    print_section("Phase 4: tenant isolation and RBAC")
    tenants = load_json("tenants.json")["tenants"]
    for tenant in tenants:
        print(f"Tenant {tenant['tenant_id']} -> plan={tenant['plan']}, quota={tenant['quota_limit']}, status={tenant['billing_status']}")


def demo_phase_5():
    print_section("Phase 5: hybrid retrieval")
    queries = [
        "What is the response SLA for premium customers?",
        "What is the enterprise billing status and quota?",
    ]
    for query in queries:
        print(f"Query: {query}")
        if "SLA" in query:
            print("Likely hit: doc-alpha-policy")
        else:
            print("Likely hit: doc-beta-finance")


def demo_phase_6():
    print_section("Phase 6: image and table retrieval")
    docs = load_json("documents.json")["documents"]
    for doc in docs:
        if doc["content_types"]:
            print(f"{doc['document_id']} -> content_types: {', '.join(doc['content_types'])}")


def demo_phase_7():
    print_section("Phase 7: evaluation and observability")
    feedback = load_json("feedback.json")["feedback"]
    usage = load_json("usage_events.json")
    print("Feedback events:")
    for item in feedback:
        print(f" - {item['feedback_id']}: score={item['score']}, comment={item['comment']}")
    print("Usage summary:")
    print(json.dumps(usage["quota_summary"], indent=2))


def demo_phase_8():
    print_section("Phase 8: public API and shareable chats")
    convs = load_json("conversations.json")["conversations"]
    for conv in convs:
        print(f"Conversation {conv['conversation_id']} shareable={conv['shareable']} -> {len(conv['messages'])} messages")


def demo_phase_9():
    print_section("Phase 9: connectors, quotas, billing, governance")
    connectors = load_json("connectors.json")["connectors"]
    for connector in connectors:
        print(f"Connector {connector['connector_id']} type={connector['type']} status={connector['status']}")
    audit = load_json("audit_events.json")["audit_events"]
    print("Audit sample:")
    for event in audit:
        print(f" - {event['action']} by {event['actor_id']} -> {event['result']}")


def main() -> None:
    print("Sample Multimodal RAG capability demo")
    print(f"Dataset root: {ROOT}")
    demo_phase_1()
    demo_phase_2()
    demo_phase_3()
    demo_phase_4()
    demo_phase_5()
    demo_phase_6()
    demo_phase_7()
    demo_phase_8()
    demo_phase_9()


if __name__ == "__main__":
    main()
