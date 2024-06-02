from homeassistant_api import Client
import secret
from datetime import datetime, timedelta


ha_ip_addr = '10.11.21.48'
entity_id = "sensor.psoc6_micropython_sensornode_f1_co2_ppm"
history_minutes = 5

with Client(
    f'http://{ha_ip_addr}:8123/api',
    secret.ha_access_token
) as client:

    # Get entity from id
    entity = client.get_entity(entity_id=entity_id)

    # Get data from this entity id for last n minutes
    start = datetime.now() - timedelta(minutes=history_minutes)
    history = client.get_entity_histories(entities=[entity], start_timestamp=start)

    # Go through each entity of the returned history data and save it's state values (here: atmospheric pressure) to a list
    for entry in history:
        values = [float(x.state) for x in entry.states]

    print(values)