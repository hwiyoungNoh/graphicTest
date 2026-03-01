//////////////////////////////////////////////////////////
// Copyright ©2015 Colorimetry Research, Inc. All Rights Reserved Worldwide. 	
// Version 1.26
//
// License: 
// This code is provided as a demonstration of the Remote Communication Software Development Kit.
// 
// This software is provided "as is" with no warranties of any kind.
// 
//////////////////////////////////////////////////////////
// CRRemote.cpp : Defines the initialization routines for the DLL.
//

#include "stdafx.h"
#include "CRRemote.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//
//TODO: If this DLL is dynamically linked against the MFC DLLs,
//		any functions exported from this DLL which call into
//		MFC must have the AFX_MANAGE_STATE macro added at the
//		very beginning of the function.
//
//		For example:
//
//		extern "C" BOOL PASCAL EXPORT ExportedFunction()
//		{
//			AFX_MANAGE_STATE(AfxGetStaticModuleState());
//			// normal function body here
//		}
//
//		It is very important that this macro appear in each
//		function, prior to any calls into MFC.  This means that
//		it must appear as the first statement within the 
//		function, even before any object variable declarations
//		as their constructors may generate calls into the MFC
//		DLL.
//
//		Please see MFC Technical Notes 33 and 58 for additional
//		details.
//

// CCRRemoteApp

BEGIN_MESSAGE_MAP(CCRRemoteApp, CWinApp)
END_MESSAGE_MAP()


// CCRRemoteApp construction

CCRRemoteApp::CCRRemoteApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}


// The one and only CCRRemoteApp object

CCRRemoteApp theApp;


// CCRRemoteApp initialization

BOOL CCRRemoteApp::InitInstance()
{
	CWinApp::InitInstance();

	return TRUE;
}
