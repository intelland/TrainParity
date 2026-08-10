from competitor_runtime import CompetitorFixture

case = CompetitorFixture("mean_of_means", fault=True)
model = case.model
case.run()
