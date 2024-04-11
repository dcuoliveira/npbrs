python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 0.95 --k 1000 --end_date "2015-12-31" --cpu_count -1
python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 0.75 --k 1000 --end_date "2015-12-31" --cpu_count -1
python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 0.50 --k 1000 --end_date "2015-12-31" --cpu_count -1
python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 0.25 --k 1000 --end_date "2015-12-31" --cpu_count -1
python -m gym.training_etfstsm --utility MaxDD --functional means --alpha 1 --k 1000 --end_date "2015-12-31" --cpu_count -1

