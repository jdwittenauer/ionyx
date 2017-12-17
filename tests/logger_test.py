from ionyx.contrib import Logger

print('Beginning logger test...')

logger = Logger('C:\\Temp\\log.txt', mode='replace')
logger.write("Testing replace mode")
logger.close()

logger = Logger('C:\\Temp\\log.txt', mode='append')
logger.write("Testing append mode")
logger.close()

print('Done.')
