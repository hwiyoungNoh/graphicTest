//////////////////////////////////////////////////////////
// Copyright ©2018 Colorimetry Research, Inc. All Rights Reserved Worldwide.
// Version 1.26
//
// License: 
// This code is provided as a demonstration of the Remote Communication Software Development Kit.
// 
// This software is provided "as is" with no warranties of any kind.
// 
//////////////////////////////////////////////////////////
#pragma once
#include "afxwin.h"


// CConnectionSetup dialog

class CConnectionSetup : public CDialog
{
	DECLARE_DYNAMIC(CConnectionSetup)

public:
	CConnectionSetup(CWnd* pParent = NULL);   // standard constructor
	virtual ~CConnectionSetup();

// Dialog Data
	enum { IDD = IDD_CONNECTIONSETUP };

	CString ChannelName(); 
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	void InitSerial();
	DECLARE_MESSAGE_MAP()
	CComboBox m_channelComboBox;
	CString m_channelName;
	afx_msg BOOL OnInitDialog();
	virtual void OnOK();
};
