# BTCML
Some ML scripts for BTC price prediction. Uses cron to run every-hour and sends an ntfy notification

# Run the prediction script every hour at the top of the hour
0 * * * * /usr/bin/python3 /home/t0fum4n/BTCML/thegoodgood.py >> /home/t0fum4n/BTCML/thegoodgood.log 2>&1

# Run the accuracy check script 5 minutes after the prediction script
5 * * * * /usr/bin/python3 /home/t0fum4n/BTCML/impvcheck.py >> /home/t0fum4n/BTCML/impvcheck.log 2>&1