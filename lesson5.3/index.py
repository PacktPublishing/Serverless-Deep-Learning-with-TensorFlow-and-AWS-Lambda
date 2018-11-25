import json

def handler(event,context):
    print('Log event',event)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello world API!')
    }