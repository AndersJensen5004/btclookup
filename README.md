# btclookup

- FRED API key is required for sharpe.py but is free to get put this in your .env file

-- use requirements.txt

- sharpe.py calculates rolling sharpe ratio on BTC
- drawdown.py calculates drawdowns v.s. S&P 500 and plots distributions, read comments at top of file
- correlation.py finds correlation between BTC and S&P500 as well as the change in correlation over time
- realized_vol.py finds the vol and vol of vol of both BTC and S&P 500 for comparasions
- model.py was created to demonstrate that ML is not useful when drawing conclusions about past price and economic data, this was unfinished becuase I did not want to tune everything, currently does not compile