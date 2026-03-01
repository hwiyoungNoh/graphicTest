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
// ConnectionSetup.cpp : implementation file
//

#include "stdafx.h"
#include "RemoteCommunication.h"
#include "ConnectionSetup.h"
#include "enumser.h"


// CConnectionSetup dialog

IMPLEMENT_DYNAMIC(CConnectionSetup, CDialog)

CConnectionSetup::CConnectionSetup(CWnd* pParent /*=NULL*/)
	: CDialog(CConnectionSetup::IDD, pParent)
	, m_channelName(_T(""))
{
}

CConnectionSetup::~CConnectionSetup()
{
}

void CConnectionSetup::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_CHANNEL, m_channelComboBox);
	DDX_CBString(pDX, IDC_COMBO_CHANNEL, m_channelName);
}


BEGIN_MESSAGE_MAP(CConnectionSetup, CDialog)

END_MESSAGE_MAP()


// CConnectionSetup message handlers

void CConnectionSetup::InitSerial()
{
	//Initialize COM (Required by CEnumerateSerial::UsingWMI)
	HRESULT hr = CoInitialize(NULL);
	if (FAILED(hr))
	{
		_tprintf(_T("Failed to initialize COM, Error:%x\n"), hr);
	}

	//Initialize COM security (Required by CEnumerateSerial::UsingWMI)
	hr = CoInitializeSecurity(NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE, NULL);
	if (FAILED(hr))
	{
		_tprintf(_T("Failed to initialize COM security, Error:%x\n"), hr);
		CoUninitialize();
	}

    m_channelComboBox.Clear();

	CUIntArray ports;
	CStringArray friendlyNames;
	CStringArray sPorts;

	if (CEnumerateSerial::UsingWMI(ports, friendlyNames))
	{
		for(int i=0; i<ports.GetSize(); i++)
		{
			CString portName;
			portName.Format(_T("COM%d"), ports.GetAt(i));
			m_channelComboBox.AddString(portName);
			if(i==0)
				m_channelName = portName;
		}
	}


   /* Dim ports As String() = SerialPort.GetPortNames()

    Dim port As String
    For Each port In ports
        Me._channelComboBox.Items.Add(port)
    Next port
    If (_channelComboBox.Items.Count > 0) Then
        _channelComboBox.SelectedIndex = 0
    End If*/

}


CString CConnectionSetup::ChannelName()
{
	return m_channelName;
}

BOOL CConnectionSetup::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO: Add your message handler code here

	InitSerial();

	UpdateData(FALSE);
	return TRUE;	// return TRUE unless you set the focus to a control	

}
void CConnectionSetup::OnOK() 
{
	UpdateData(TRUE);
	CDialog::OnOK();
}
