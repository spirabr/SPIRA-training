# SPIRA Training

A training pipeline to [SPIRA](https://spira.ime.usp.br/) project.

## Development

SPIRA-training is built using Python and uses Docker for containerization.

To run the project on the host machine, poetry is required. 

### Installing dependencies

It's suggested to run the project with [Docker](https://docs.docker.com/engine/install/).
Alternatively, [install poetry](https://python-poetry.org/docs/#installation), then install the project packages:

```bash
# at the root
poetry install
```

## Execution 

To execute the project, run:

```sh
# at the root
make app
```

Alternatively, execute:

```sh
# at the root
poetry run python src/spira_training/main.py
```

## Tests

To execute the tests, run:

```sh
# at the root
make unit-tests
```

Alternatively, execute: 

```sh
# at the root
poetry run pytest
```

## Authors
- [Roberto Bolgheroni](https://github.com/bolgheroni)
- [Lucas Quaresma](https://github.com/lucasqml)

## Acknowledgements
- [Renato Cordeiro Ferreira](https://linktr.ee/renatocf)
