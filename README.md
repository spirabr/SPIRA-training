# SPIRA-training

A training pipeline to [SPIRA](https://spira.ime.usp.br/) project.

## Development

SPIRA-training is built using Python and uses Docker for containerization.

To run the project on the host machine, poetry is required. 

### Installing dependencies

If you prefer to run the project locally, install poetry, then install the project packages:

```bash
poetry install
```

## Compilation

The compilation of all Python files is handled by Docker.

To build the Docker image, run:

```sh
docker compose build
```

## Execution 

To execute the project, run:

```sh
docker compose up
```

Alternatively, to run locally, execute:

```sh
poetry run python src/spira_training/main.py
```

## Tests

The tests are run in the build phase.
To locally run them, execute: 

```sh
poetry run pytest
```
## Authors
- [Roberto Bolgheroni](https://github.com/bolgheroni)
- [Lucas Quaresma](https://github.com/lucasqml)

## Acknowledgements
- [Renato Cordeiro Ferreira](https://linktr.ee/renatocf)
