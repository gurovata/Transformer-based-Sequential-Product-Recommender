# Transformer-based Sequential Product Recommender

Это мой учебный ML-проект про рекомендательные системы.

## Идея проекта

Проект имитирует каталог цифровых продуктов: мобильная связь, интернет, ТВ, финансы, развлечения, облачные сервисы, устройства и travel-сервисы.

У пользователя есть история действий:

```text
view -> cart -> purchase
```

Задача модели:

```text
предыдущие товары пользователя -> следующий товар
```

То есть если пользователь уже взаимодействовал с несколькими продуктами, система должна выдать top-K товаров, которые могут быть ему интересны дальше.

## Что есть в проекте

- генератор синтетических данных;
- подготовка пользовательских последовательностей;
- split на `train / validation / test` через `leave-two-out`;
- baseline-модели: `Popularity`, `ItemKNN`, `ALS/SVD`;
- Transformer-модель в стиле SASRec;
- ranking-метрики: `Recall@K`, `HitRate@K`, `NDCG@K`, `MRR@K`;
- простая EDA-часть;
- SQLite-схема и примеры SQL-запросов;
- FastAPI endpoint для рекомендаций;
- Docker-файлы;
- тесты для метрик и разбиения последовательностей.

## Как я работала с данными

Так как реального датасета в проекте нет, я сделала генератор синтетических данных. Он создает:

- пользователей;
- товары;
- категории товаров;
- ценовые сегменты;
- временные события;
- разные типы действий: `view`, `cart`, `purchase`.

Я не делала данные полностью случайными. В них есть простые паттерны:

- у пользователей есть любимые категории;
- некоторые категории часто идут друг за другом;
- часть товаров заметно популярнее остальных;
- у пользователей разная длина истории;
- события отсортированы по времени.

После генерации данные превращаются в последовательности товаров для каждого пользователя. Для обучения используется такая логика:

```text
train:      вся история, кроме двух последних товаров
validation: предпоследний товар
test:       последний товар
```

Внутри модели используются специальные id:

- `PAD = 0`;
- `MASK = 1`;
- реальные товары начинаются с `2`.

## Модели

Я сравнила несколько подходов, чтобы Transformer не был единственной точкой отсчета.

| Модель | Что делает |
| --- | --- |
| Popularity | Рекомендует самые популярные товары. |
| ItemKNN | Ищет похожие товары через cosine similarity. |
| ALS/SVD | Делает collaborative filtering через матричное разложение. |
| Transformer | Учитывает порядок товаров в истории пользователя. |

Transformer реализован как небольшая SASRec-style модель:

- item embeddings;
- positional embeddings;
- category embeddings;
- causal TransformerEncoder;
- предсказание следующего item'а.

Я оставила модель маленькой, чтобы она спокойно обучалась на CPU.

## Результаты

Результаты на синтетическом датасете из default config:

- пользователей: 450;
- товаров: 260;
- взаимодействий: 9,423;
- средняя длина истории: 20.94;
- sparsity: 0.9195.

Метрики на test split:

| Модель | Recall@5 | NDCG@5 | MRR@5 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.0956 | 0.0610 | 0.0498 | 0.2356 | 0.0989 |
| ItemKNN | 0.0978 | 0.0624 | 0.0510 | 0.2689 | 0.1100 |
| ALS/SVD | 0.0356 | 0.0175 | 0.0116 | 0.1222 | 0.0419 |
| Transformer | 0.0956 | 0.0619 | 0.0511 | 0.2400 | 0.1012 |

Мой вывод: на этих синтетических данных лучше всего сработал `ItemKNN`, потому что в данных есть довольно прямые co-occurrence-паттерны. Transformer не победил все baseline'ы, но получился конкурентным и лучше popularity по `MRR@5`.

## Как запустить

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить весь pipeline:

```bash
python -m src.data.generate_synthetic
python -m src.data.prepare_sequences
python -m src.features.item_features
python -m src.evaluation.eda
python -m src.data.load_sqlite
python -m src.training.train_baselines
python -m src.training.train_transformer
python -m src.evaluation.evaluate --models popularity itemknn als transformer --split test
python -m pytest
```

Запустить API:

```bash
python -m src.api.app
```

Swagger UI:

```text
http://localhost:8000/docs
```

Пример запроса:

```json
{
  "user_id": 1,
  "history": [10, 25, 107],
  "top_k": 5
}
```

Пример ответа:

```json
{
  "recommendations": [
    {"item_id": 238, "score": 1.8489},
    {"item_id": 211, "score": 1.7198},
    {"item_id": 184, "score": 1.7090}
  ],
  "model": "transformer"
}
```

## Структура

```text
src/data        генерация и подготовка данных
src/models      baseline-модели и Transformer
src/training    обучение моделей
src/evaluation  метрики, EDA и оценка
src/api         FastAPI-сервис
sql             SQLite schema
tests           тесты
configs         конфигурация
```

## Что можно улучшить

Если продолжать проект, я бы добавила:

- реальный датасет вместо синтетики;
- CatBoost baseline с negative sampling;
- BERT4Rec-style masked item modeling;
- разбор качества по сегментам пользователей;
- больше тестов для подготовки данных и API.
