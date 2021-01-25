#! /bin/bash
docker container stop $(docker container ps -aq --filter "ancestor=sanluosizhou/selfdl:mopo")