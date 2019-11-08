# Docker Image #

## To Build ##

```
docker build -t adeelhazv/aws-batch-ml .
```

## To Test ##

```
docker run -it --rm --runtime=nvidia -v $HOME/.aws:/root/.aws:ro adeelhazv/aws-batch-ml bash
```

# Job Definition #

Edit [`job-definition.json`](job-definition.json) prior to use.

```
aws batch register-job-definition --cli-input-json file://job-definition.json
```
