python -m gym.training_etfstsm --utility Sharpe --functional means --alpha 0.95 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility AvgDD --functional means --alpha 0.95 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 0.95 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility "% Positive Ret." --functional means --alpha 0.95 --k 100 --end_date "2015-12-31"

python -m gym.training_etfstsm --utility Sharpe --functional means --alpha 0.75 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility AvgDD --functional means --alpha 0.75 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 0.75 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility "% Positive Ret." --functional means --alpha 0.75 --k 100 --end_date "2015-12-31"

python -m gym.training_etfstsm --utility Sharpe --functional means --alpha 1 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility AvgDD --functional means --alpha 1 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 1 --k 100 --end_date "2015-12-31"
python -m gym.training_etfstsm --utility "% Positive Ret." --functional means --alpha 1 --k 100 --end_date "2015-12-31"

