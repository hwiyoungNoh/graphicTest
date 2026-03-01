#include "StdAfx.h"
#include "Channel.h"

CChannel::CChannel(void)
{
	m_pNotifier = NULL;
}

CChannel::~CChannel(void)
{

}

BOOL CChannel::IsOpen() const
{
	return FALSE;
}

int CChannel::ReadTimeout()
{
	return 0;
}

void CChannel::SetReadTimeout(int timeout)
{
}

void CChannel::Open()
{
}

void CChannel::Close()
{
}


void CChannel::WriteLine(const CString& text)
{
}

CString CChannel::ReadExisting()
{
	return CString();
}

void CChannel::SetNotifier(INotifier *notifier)
{
	m_pNotifier = notifier;
}

void CChannel::NotifyCaller(NotificationType type, const CString& buffer)
{
	if(m_pNotifier)
	{
		 m_pNotifier->OnNotifcation(type, buffer);
	}
}