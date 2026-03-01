#pragma once

#ifndef __CHANNEL_H__
#define __CHANNEL_H__

#include "INotifier.h"

#ifndef CCHANNEL_EXT_CLASS
#define CCHANNEL_EXT_CLASS
#endif

class CCHANNEL_EXT_CLASS CChannel
{
public:
	CChannel(void);
	virtual ~CChannel(void);

	virtual BOOL IsOpen() const = 0;
	virtual int ReadTimeout() = 0;
	virtual void SetReadTimeout(int timeout) = 0;

	virtual void Open() = 0;
	virtual void Close() = 0;

	virtual void WriteLine(const CString& text) = 0;
	virtual CString ReadExisting() = 0;

	void SetNotifier(INotifier* notifier);
protected:
	void NotifyCaller(NotificationType type, const CString& buffer);
private:
	INotifier* m_pNotifier;
};

#endif //__CHANNEL_H__