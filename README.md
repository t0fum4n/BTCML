# BTCML
Some ML scripts for BTC price prediction

# Run the prediction script at the start of every hour
0 * * * * /usr/bin/python3 /path/to/your/thegoodgood.py

# Run the accuracy check script 5 minutes after the prediction script
5 * * * * /usr/bin/python3 /path/to/your/check.py
