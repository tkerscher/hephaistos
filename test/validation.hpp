#pragma once

extern bool _errorFlag; //defined in main
inline bool hasValidationErrorOccurred() {
	//clear flag and return its original value
	bool val = _errorFlag;
	_errorFlag = false;
	return val;
}
