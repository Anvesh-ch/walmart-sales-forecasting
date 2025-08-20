.PHONY: all clean prepare_data train_prophet train_xgb train_sarimax backtest_all compare_all export_dashboard

all: prepare_data train_prophet train_xgb backtest_all compare_all export_dashboard

clean:
	rm -rf data/interim data/processed data/artifacts exports

prepare_data:
	python -m src.pipelines.prepare_data --seed 42

train_prophet:
	python -m src.pipelines.train_prophet --top_n 50 --seed 42

train_xgb:
	python -m src.pipelines.train_xgb --seed 42

train_sarimax:
	python -m src.pipelines.train_sarimax --enable false

backtest_all:
	python -m src.pipelines.backtest_all --folds 4 --horizon 13 --seed 42

compare_all:
	python -m src.pipelines.compare_all

export_dashboard:
	python -m src.pipelines.export_dashboard
