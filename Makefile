app:
	docker compose up app

unit-tests: 
	docker compose run --rm tests
