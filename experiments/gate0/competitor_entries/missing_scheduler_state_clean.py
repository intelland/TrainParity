from competitor_runtime import CompetitorFixture

case = CompetitorFixture("missing_scheduler_state", fault=False)
model = case.model
case.run()
