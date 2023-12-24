# Vehicle Value Estimator
AI-based solution for predicting car prices, leveraging machine learning to analyse vehicle features for accurate valuations.

## Project Description
This project focuses on developing a robust regression model that predicts car sales prices. At its core is a comprehensive car sale adverts dataset, courtesy of AutoTrader UK. This dataset encompasses various anonymised vehicle adverts, providing insights into attributes like make, model, colour, mileage, and corresponding selling prices.

## Setup
```
git clone https://github.com/ade-mola/vehicle-value-estimator.git
cd vehicle-value-estimator
mamba env create -f conda.yaml --solver libmamba
poetry install
```

## Clean Code
```
make fix
make lint
```