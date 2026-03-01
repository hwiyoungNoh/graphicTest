// CRRemote.h : main header file for the CRRemote DLL
//

#pragma once

#ifndef __CCRREMOTE_H__
#define __CCRREMOTE_H__

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CCRRemoteApp
// See CRRemote.cpp for the implementation of this class
//

class CCRRemoteApp : public CWinApp
{
public:
	CCRRemoteApp();

// Overrides
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
};

#endif //__CCRREMOTE_H__