import boto3

def handler(event,context):
	for message in event['Records']:
		client = boto3.client('sqs')
		sqsAddress = client.get_queue_url(QueueName='WriteSQS')
		response = client.send_message(QueueUrl=sqsAddress['QueueUrl'],MessageBody=message['body'])
	return