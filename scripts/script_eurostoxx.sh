#!/bin/bash

cd src
python3 main.py --strategy all --quote ACA.PA   --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.03 --nn-days 10 --nn-epochs 300
python3 main.py --strategy all --quote AGN.AS   --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.03 --nn-days 10 --nn-epochs 300
python3 main.py --strategy all --quote ALV.DE   --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.03 --nn-days 10 --nn-epochs 300
python3 main.py --strategy all --quote BLND.L   --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.05 --nn-days 20 --nn-epochs 300
python3 main.py --strategy all --quote BME.MC   --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.03 --nn-days 10 --nn-epochs 300
python3 main.py --strategy all --quote DB1.DE   --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.05 --nn-days 20 --nn-epochs 300
python3 main.py --strategy all --quote SAB.MC   --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.03 --nn-days 20 --nn-epochs 300
python3 main.py --strategy all --quote SAMPO.HE --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.03 --nn-days 10 --nn-epochs 300
python3 main.py --strategy all --quote SAN      --from-date 2011-12-22 --to-date 2013-12-22 --nn-gain 0.07 --nn-loss 0.05 --nn-days 10 --nn-epochs 300
