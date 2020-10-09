# Configuration using dict method
# To unpack:
# On main script -> ex: config.PETS2009.get('distance')
PETS2009 = {
	'distance': 70,
	'width': 768,
	'height': 576
}

TOWNCENTRE = {
	'distance': 68.5,
	'width': 1280,
	'height': 720
}

VIRAT = {
	'distance': 55, 
	'width': 1280,
	'height': 720
}

COLORS = {
	'GREEN' : (0,255,0),
	'RED' : (0,0,255),
	'YELLOW' : (0,255,255),
	'WHITE' : (255,255,255),
	'ORANGE' : (0,165,255),
	'BLUE' : (255,0,0),
	'GREY' :(192,192,192)
}

# Configuration using class method
class Config:

	'''
	Usage; to unpack on main script:
	
	from config import Config
	param1, param2, param3 = Config.get2Data(variable)

	'''
	def get2Data(videoname):
		if videoname== 'TownCentre.mp4':
			distance = 68.5
			width = 1280
			height = 720
		if videoname== 'PETS2009.mp4':
			distance = 70
			width = 768
			height = 576
		if videoname== 'VIRAT.mp4':
			distance = 55
			width == 1280
			height == 720
		
		return distance,width,height


	'''
	Usage; to unpack on main script:
	
	from config import Config
	param = Config.colors(variable)

	'''
	def colors(color):
		if color == 'GREEN':
			return (0,255,0)

		if color == 'RED':
			return (0,0,255)

		if color == 'YELLOW':
			return (0,255,255)

		if color == 'WHITE':
			return (255,255,255)

		if color == 'ORANGE':
			return (0,165,255)

		if color == 'BLUE':
			return (255,0,0)

		if color == 'GREY':
			return (192,192,192)