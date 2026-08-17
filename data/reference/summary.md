# TAS 1.6 Reference Data Summary

This folder carries the authors' **TAS 1.6** (Telecare Assistance Service) replication dump used as ground truth for the `experiment` method. The data ships in three QoS scenario folders (`Cost-QoS/`, `Preferred-QoS/`, `Reliability-QoS/`), each with `invocations.csv`, `log.csv`, `results.csv` and eight plots. This file documents what the CSV columns mean.

## Context

Based on the TAS context and the CSV structure, the per-row schema captures one **service invocation** inside a larger telecare workflow:

- **Request ID** — sequential number identifying each service request (e.g. 1-391 in the sample dumps).
- **Service Name** — the specific service invoked, one of:
  - `MedicalService1/2/3` — medical monitoring services (QoS variants).
  - `AlarmService1/2/3` — alarm notification services (QoS variants).
  - `DrugService` — medication management service.
  - `AssistanceService` — the final coordination service (end of workflow).
- **Success Status** — boolean indicating whether the invocation succeeded.
- **Service Cost / Time** — per-invocation cost or execution time (depends on service type).
- **Timestamp (Start)** — when the invocation started, in seconds from simulation start.
- **Response Time** — actual response time experienced for the invocation, in seconds.
- **Total Request Cost / Time** — cumulative cost or time for the whole request, only on the final `AssistanceService` row.

## QoS scenarios

The three folders under `data/reference/` differ in which services get selected under their adaptation policy:

- **Cost-QoS** optimises for lowest cost.
- **Preferred-QoS** optimises for a balanced mix of cost, reliability, and performance.
- **Reliability-QoS** optimises for the highest success rate / availability.

Columns 4 (service cost) and 6 (response time) show the most scenario-to-scenario variation; the underlying service identities shift accordingly.

## Column profiles per scenario

### Cost QoS

| Column # | Name | Description |
|---|---|---|
| **Column 1** | `Request ID` | Sequential identifier for each service request (1, 2, 3, ...). |
| **Column 2** | `Service Name` | Specific service invoked. `MedicalService3` (medical monitoring), `DrugService` (medication), `AlarmService3` (alarm notification), `AssistanceService` (final coordination, end of workflow). |
| **Column 3** | `Success Status` | Boolean indicating invocation success (`true` in all rows shown). |
| **Column 4** | `Service Cost` | Cost per invocation (constant per service type). `MedicalService3`: 2.15; `DrugService`: 5.0; `AlarmService3`: 2.0; `AssistanceService`: varies (7.15, 4.15, or 2.0 depending on workflow). |
| **Column 5** | `Arrival Timestamp` | Absolute timestamp (seconds, from simulation start) when the service request arrived at that specific service. |
| **Column 6** | `Response Time` | Actual response time (seconds) for the invocation (queue wait + processing). |
| **Column 7** | `Total Request Cost` | Cumulative workflow cost; only present on the final `AssistanceService` row. |

### Preferred QoS

| Column # | Name | Description |
|---|---|---|
| **Column 1** | `Request ID` | Sequential identifier for each service request. Each request may invoke multiple services. |
| **Column 2** | `Service Name` | Specific service invoked. Includes `MedicalService1/2/3`, `AlarmService1/2/3`, `DrugService`, `AssistanceService` (end of workflow). |
| **Column 3** | `Success Status` | Boolean indicating invocation success (`true` in all rows shown). |
| **Column 4** | `Service Cost` | Cost per invocation (constant per service type). `MedicalService3`: 2.15; `AlarmService3`: 2.0; `DrugService`: 5.0; `AssistanceService`: varies by workflow path. |
| **Column 5** | `Arrival Timestamp` | Absolute timestamp (seconds, from simulation start) when the request arrived at that specific service. Increases monotonically as the request progresses through the workflow. |
| **Column 6** | `Response Time` | Actual response time (seconds) for the invocation (queue wait + processing). Key performance metric. |
| **Column 7** | `Total Request Cost` | Cumulative workflow cost, only on the final `AssistanceService` row; sums the service costs in the workflow. |

### Reliability QoS

| Column # | Name | Description |
|---|---|---|
| **Column 1** | `Request ID` | Sequential identifier. Each request typically invokes 2-3 services. |
| **Column 2** | `Service Name` | Specific service invoked. `MedicalService2` (Reliability variant, cost 14.0), `AlarmService2` (Reliability variant, cost 12.0), `DrugService` (cost 5.0), `AssistanceService` (end of workflow). |
| **Column 3** | `Success Status` | Boolean indicating invocation success (`true` in all rows shown; expected for Reliability QoS). |
| **Column 4** | `Service Cost` | Cost per invocation. `MedicalService2`: 14.0 (higher cost, more reliable); `AlarmService2`: 12.0 (higher cost, more reliable); `DrugService`: 5.0; `AssistanceService`: total workflow cost (19.0, 12.0, or 26.0). |
| **Column 5** | `Arrival Timestamp` | Absolute timestamp (seconds, from simulation start ~118738 seconds). Increases as the request progresses through the workflow. |
| **Column 6** | `Response Time` | Actual response time (seconds), includes queue wait + processing. Key performance metric. |
| **Column 7** | `Total Request Cost` | Cumulative workflow cost, only on the final `AssistanceService` row. |
