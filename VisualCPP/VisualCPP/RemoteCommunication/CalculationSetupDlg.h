#pragma once


#include "CRColorimeter.h"

// CCalculationSetupDlg dialog

class CCalculationSetupDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CCalculationSetupDlg)

public:
	CCalculationSetupDlg(CWnd* pParent = NULL);   // standard constructor
	virtual ~CCalculationSetupDlg();

// Dialog Data
	enum { IDD = IDD_CALCULATIONSETUP };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
	BOOL OnInitDialog();

public:
	void SetColorimeter(CCRColorimeter* pColorimeter);
	void EnableSettings(BOOL enable);
	void UpdateSettings(BOOL enable);

public:
	float m_fWhitex;
	float m_fWhitey;
	CCRColorimeter* m_pColorimeter;
	
	CSpinButtonCtrl m_whitexSpinButtonCtrl;
	CSpinButtonCtrl m_whiteySpinButtonCtrl;
	
	CEdit m_whitexEdit;
	CEdit m_whiteyEdit;
	afx_msg void OnEnChangeEditWhitex();
};
