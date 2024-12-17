class Logger:
	def __init__(self, enable_logging: bool = True) -> None:
		self._enable_logging: bool = enable_logging

	def set_logging(self, enable_logging: bool) -> None:
		self._enable_logging = enable_logging

	def __call__(self, *args):
		if self._enable_logging:
			print(*args)
