#pragma once


#ifndef __INOTIFIER_H__
#define __INOTIFIER_H__

enum NotificationType
{
	DataRead,
	DataWritten,
};


class INotifier
{
public:
virtual void OnNotifcation(NotificationType type, const CString& buffer) = 0;
};


#endif //__INOTIFIER_H__