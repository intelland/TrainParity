from competitor_runtime import CompetitorFixture

case = CompetitorFixture("sample_duplication", fault=True)
model = case.model
case.run()
