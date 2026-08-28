.PHONY: help up down logs schema reset seed fresh payment ui eval report

help:
	@echo "up        bring up n8n and mailpit"
	@echo "down      stop containers"
	@echo "logs      tail container logs"
	@echo "schema    apply schema.sql to the database"
	@echo "reset     truncate app tables (keeps schema)"
	@echo "seed      load the corpus into the claims table"
	@echo "fresh     reset then seed"
	@echo "payment   run the payment stub endpoint"
	@echo "ui        run the streamlit app"
	@echo "eval      run the evaluation suite"
	@echo "report    print the coverage report"

up:
	docker compose up -d
	@echo "n8n      http://localhost:5678"
	@echo "mailpit  http://localhost:8025"

down:
	docker compose down

logs:
	docker compose logs -f

schema:
	psql "$$PG_URL" -f db/schema.sql

reset:
	python scripts/reset.py

seed:
	python scripts/seed.py

fresh: reset seed

payment:
	uvicorn services.payment_stub:app --host 0.0.0.0 --port 8100

ui:
	streamlit run ui/app.py

eval:
	pytest evals/ -v

report:
	python scripts/report.py
