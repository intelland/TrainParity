from competitor_runtime import CompetitorFixture

case = CompetitorFixture("missing_rng_state", fault=True)
model = case.model
case.run()
