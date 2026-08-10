from competitor_runtime import CompetitorFixture

case = CompetitorFixture("missing_rng_state", fault=False)
model = case.model
case.run()
