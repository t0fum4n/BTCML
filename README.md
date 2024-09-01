# BTCML
Some ML scripts for BTC price prediction

# Run the prediction script every hour at the top of the hour
0 * * * * /usr/bin/python3 /home/#####/BTCML/thegoodgood.py

# Run the accuracy check script 5 minutes after the prediction script
5 * * * * /usr/bin/python3 /home/#####/BTCML/check.py

