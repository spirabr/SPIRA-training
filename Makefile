app:
	docker compose up app

unit-tests: 
	docker compose run --rm tests

type-check:
	poetry run pyright

MAIN_DIFF=$(shell git diff main --diff-filter=ACM --name-only | grep ".py") 

type-check-main:	
	poetry run pyright $(MAIN_DIFF)